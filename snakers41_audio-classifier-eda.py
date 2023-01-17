import gc

import os

import librosa

import numpy as np

import pandas as pd

from glob import glob

from pathlib import Path

from librosa import display

import matplotlib.pyplot as plt

from scipy.io.wavfile import read

from IPython.display import HTML, Audio, display_html

from IPython.display import display as display_ipython



pd.set_option('display.max_colwidth', 500)
label_dict = {

    'speech': 0,

    'music': 1,

    'noise': 2    

}
data_folder = '../../kaggle/input/silero-audio-classifier'
def read_audio(path):

    sr, wav = read(path)

    assert sr == 16000

    assert len(wav) == 16000 * 3

    assert len(wav.shape) == 1

    return wav





def read_audio_norm(path):

    wav = read_audio(path)

    abs_max = np.abs(wav).max()

    wav = wav.astype('float32')

    if abs_max > 0:

        wav *= 1 / abs_max

    return wav





def audio_player(audio_path):

    return f'<audio preload="none" controls="controls"><source src="{audio_path}" type="audio/wav"></audio>'





def display_manifest(df):

    display_df = df

    display_df['wav'] = [audio_player(path) for path in display_df.wav_path]

    audio_style = '<style>audio {height:44px;border:0;padding:0 20px 0px;margin:-10px -20px -20px;}</style>'

    display_df = display_df[['wav', 'label']]

    display_ipython(HTML(audio_style + display_df.to_html(escape=False)))

    del display_df

    gc.collect()
train_path = Path(data_folder) / 'train'

train_df = pd.read_csv(Path(data_folder) / 'train.csv')

train_df['target'] = train_df['label'].apply(lambda x: label_dict.get(x, -1))



assert set([os.path.relpath(path, data_folder + '/train') 

            for path 

            in glob(f'{train_path}/train/*/*.wav')]) == set(train_df.wav_path.values)



train_df['wav_path'] = train_df['wav_path'].apply(lambda x: str(Path(data_folder) / Path('train') / x))
train_df.label.value_counts().values

wav = read_audio(train_df.sample(n=1).iloc[0].wav_path)

Audio(wav, rate=16000, autoplay=True)
window_size = 0.02

window_stride = 0.01

sample_rate = 16000



n_fft = int(sample_rate * (window_size + 1e-8))

win_length = n_fft

hop_length = int(sample_rate * (window_stride + 1e-8))



kwargs = {

    'n_fft': n_fft,

    'hop_length': hop_length,

    'win_length': n_fft

}



def stft(wav):

    D = librosa.stft(wav,

                     **kwargs)

    mag, phase = librosa.magphase(D)    

    return mag
def visualize_spect(spects,

                    labels):

    assert len(spects) == 3

    assert len(labels) == 3

    

    plt.figure(figsize=(20, 10))



    plt.subplot(3, 3, 1)

    librosa.display.specshow(spects[0], y_axis='log')

    plt.colorbar(format='%+2.0f dB')

    plt.title(f'Log-frequency power spectrogram, {labels[0]}')



    plt.subplot(3, 3, 2)

    librosa.display.specshow(spects[1], y_axis='log')

    plt.colorbar(format='%+2.0f dB')

    plt.title(f'Log-frequency power spectrogram, {labels[1]}')



    plt.subplot(3, 3, 3)

    librosa.display.specshow(spects[2], y_axis='log')

    plt.colorbar(format='%+2.0f dB')

    plt.title(f'Log-frequency power spectrogram, {labels[2]}')



    plt.tight_layout()

    plt.show()
spects = []

labels = []



_ = ['speech', 'music', 'noise']



for i in range(0, 3):

    s = train_df[train_df.label == _[i]].sample(n=1).iloc[0]

    spects.append(stft(read_audio_norm(s.wav_path)))

    labels.append(s.label)

    

visualize_spect(spects,

                labels)    