import argparse
import re
import logging
import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
import ONNXVITS_infer
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch
import openai
import os
import warnings
from pydub import AudioSegment
import pyaudio

logging.getLogger('numba').setLevel(logging.WARNING)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")

openai.api_key = "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"

chat_content = "The following is a conversation with an AI assistant. " \
               "The assistant is helpful, creative, clever, and very friendly.\n\n" \
               "Human: Hello, who are you?\n" \
               "AI: I am Norma, an AI created by OpenAI. How can I help you today?\n" \
               "Human: From now use English please.\n"\
               "AI: OK!\n"

sound_list = []
text = ""

stop_count = 0

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}

limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces


def transcribe(audio):

    global chat_content
    global sound_list
    global text
    global stop_count

    y, sr = librosa.load(audio)

    # 计算RMS能量
    rms = librosa.feature.rms(y=y)

    # 判断RMS能量是否超过阈值
    threshold = 0.1
    if rms.max() > threshold:
        print('该音频文件中包含明显语音信号')

        sound_list.append(AudioSegment.from_wav(audio))
        stop_count = 0
        return text

    else:
        stop_count = stop_count + 1
        if sound_list:
            if stop_count > 5:
                combined_sound = None
                for sound in sound_list:
                    if combined_sound:
                        combined_sound = combined_sound + sound
                    else:
                        combined_sound = sound
                combined_sound.export("combined_audio.wav", format="wav")
                sound_list = []

                with open("combined_audio.wav", "rb") as audio_file:
                    transcript = str(openai.Audio.transcribe("whisper-1", file=audio_file)["text"])
                    print(transcript)
                if transcript != "":
                    chat_content = chat_content + "Human: %s\nAI:" % transcript

                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=chat_content,
                        temperature=0.9,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0.0,
                        presence_penalty=0.6,
                        stop=[" Human:", " AI:"]
                    )

                    chat_content = chat_content + response["choices"][0]["text"] + "\n"

                    text = response["choices"][0]["text"]
                    print(text)

                    text = language_marks["English"] + text + language_marks["English"]
                    speaker_id = 0
                    stn_tst = get_text(text, hps)
                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0).to(device)
                        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                        sid = LongTensor([speaker_id]).to(device)
                        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                            length_scale=1.0 / 1.0)[0][0, 0].data.cpu().float().numpy()
                    p = pyaudio.PyAudio()

                    # 打开输出流
                    stream = p.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=20000,
                                    output=True)

                    # 播放音频
                    stream.write(audio.astype(np.float32).tobytes())

                    # 关闭流和 PyAudio
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    del stn_tst, x_tst, x_tst_lengths, sid
                    stop_count = 0
                    return text
            else:
                sound_list.append(AudioSegment.from_wav(audio))
                return text
        return text


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(speaker, language, speed):
        global text
        if limitation:
            text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
            max_len = 150
            if text_len > max_len:
                return "Error: Text is too long", None
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()

        del stn_tst, x_tst, x_tst_lengths, sid

        # return text, (hps.data.sampling_rate, audio)
        return audio

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, input_audio):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        if limitation and duration > 30:
            return "Error: Audio is too long", None
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio).to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.symbols, [] if 0 else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_text):
        return (_clean_text(input_text, hps.data.text_cleaners), input_text) if is_symbol_input \
            else (temp_text, temp_text)

    return to_symbol_fn


models_tts = []
models_vc = []
models_info = [
    {
        "title": "Trilingual",
        "languages": ['日本語', '简体中文', 'English', 'Mix'],
        "description": """
    This model is trained on a mix up of Umamusume, Genshin Impact, Sanoba Witch & VCTK voice data to learn multilanguage.
    All characters can speak English, Chinese & Japanese.\n\n
    To mix multiple languages in a single sentence, wrap the corresponding part with language tokens
     ([JA] for Japanese, [ZH] for Chinese, [EN] for English), as shown in the examples.\n\n
    这个模型在赛马娘，原神，魔女的夜宴以及VCTK数据集上混合训练以学习多种语言。
    所有角色均可说中日英三语。\n\n
    若需要在同一个句子中混合多种语言，使用相应的语言标记包裹句子。
    （日语用[JA], 中文用[ZH], 英文用[EN]），参考Examples中的示例。
    """,
        "model_path": "./pretrained_models/G_trilingual.pth",
        "config_path": "./configs/uma_trilingual.json",
        "examples": [['你好，训练员先生，很高兴见到你。', '草上飞 Grass Wonder (Umamusume Pretty Derby)', '简体中文', 1, False],
                     ['To be honest, I have no idea what to say as examples.', '派蒙 Paimon (Genshin Impact)', 'English',
                      1, False],
                     ['授業中に出しだら，学校生活終わるですわ。', '綾地 寧々 Ayachi Nene (Sanoba Witch)', '日本語', 1, False],
                     ['[JA]こんにちわ。[JA][ZH]你好！[ZH][EN]Hello![EN]', '綾地 寧々 Ayachi Nene (Sanoba Witch)', 'Mix', 1, False]],
        "onnx_dir": "./ONNX_net/G_trilingual/"
    },
    {
        "title": "Japanese",
        "languages": ["Japanese"],
        "description": """
                       This model contains 87 characters from Umamusume: Pretty Derby, Japanese only.\n\n
                       这个模型包含赛马娘的所有87名角色，只能合成日语。
                       """,
        "model_path": "./pretrained_models/G_jp.pth",
        "config_path": "./configs/uma87.json",
        "examples": [['お疲れ様です，トレーナーさん。', '无声铃鹿 Silence Suzuka (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['張り切っていこう！', '北部玄驹 Kitasan Black (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['何でこんなに慣れでんのよ，私のほが先に好きだっだのに。', '草上飞 Grass Wonder (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['授業中に出しだら，学校生活終わるですわ。', '目白麦昆 Mejiro Mcqueen (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['お帰りなさい，お兄様！', '米浴 Rice Shower (Umamusume Pretty Derby)', 'Japanese', 1, False],
                     ['私の処女をもらっでください！', '米浴 Rice Shower (Umamusume Pretty Derby)', 'Japanese', 1, False]],
        "onnx_dir": "./ONNX_net/G_jp/"
    },
]

download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#tts-audio").querySelector("audio");
    let text = root.querySelector("#input-text").querySelector("textarea");
    if (audio == undefined)
        return;
    text = text.value;
    if (text == undefined)
        text = Math.floor(Math.random()*100000000);
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = 'microphone.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    for info in models_info[0:1]:
        name = info['title']
        lang = info['languages']
        examples = info['examples']
        config_path = info['config_path']
        model_path = info['model_path']
        description = info['description']
        onnx_dir = info["onnx_dir"]
        hps = utils.get_hparams_from_file(config_path)
        model = ONNXVITS_infer.SynthesizerTrn(
            len(hps.symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            ONNX_dir=onnx_dir,
            **hps.model).to(device)
        utils.load_checkpoint(model_path, model, None)
        model.eval()
        speaker_ids = hps.speakers
        speakers = list(hps.speakers.keys())
        models_tts.append((name, description, speakers, lang, examples,
                           hps.symbols, create_tts_fn(model, hps, speaker_ids),
                           create_to_symbol_fn(hps)))
        models_vc.append((name, description, speakers, create_vc_fn(model, hps, speaker_ids)))
    gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(source="microphone", type="filepath", streaming=True),
        ],
        outputs=[
            "textbox",
        ],
        live=True).launch()
    # app = gr.Blocks()
    # with app:
    #     gr.Markdown(
    #         "# <center> Welcome to use Norma.\n"
    #     )
    #     for i, (name, description, speakers, lang, example, symbols, tts_fn, to_symbol_fn) in enumerate(
    #             models_tts[0:1]):
    #         with gr.Row():
    #             with gr.Column():
    #                 input_audio = gr.Audio(source="microphone", type="filepath", elem_id=f"tts-audio")
    #                 char_dropdown = gr.Dropdown(choices=speakers, value=speakers[94], label='character')
    #                 language_dropdown = gr.Dropdown(choices=lang, value=lang[2], label='language')
    #                 duration_slider = gr.Slider(minimum=0.1, maximum=5, value=0.8, step=0.1,
    #                                             label='速度 Speed')
    #             with gr.Column():
    #                 text_output = gr.Textbox(label="Output Message", value="Welcome to use Norma.",
    #                                         elem_id=f"input-text")
    #                 audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
    #                 btn1 = gr.Button(value="Submit audio")
    #                 btn2 = gr.Button(value="Start to chat")
    #             btn1.click(None, inputs=[], outputs=[], _js=download_audio_js.format())
    #             btn2.click(tts_fn, inputs=[char_dropdown, language_dropdown, duration_slider],
    #                        outputs=[text_output, audio_output])
    # app.queue(concurrency_count=3).launch(show_api=False, share=args.share)
