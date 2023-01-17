import os

from pathlib import Path

import pandas as pd



import numpy as np

import librosa

import librosa.core

import librosa.feature



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from scipy import signal

import warnings

warnings.filterwarnings("ignore")



sns.set_palette("husl")



def file_load(wav_name, mono=False):

    """

    load .wav file.



    wav_name : str

        target .wav file

    sampling_rate : int

        audio file sampling_rate

    mono : boolean

        When load a multi channels file and this param True, the returned data will be merged for mono data



    return : np.array( float )

    """

    try:

        return librosa.load(wav_name, sr=None, mono=mono)

    except:

        print("file_broken or not exists!! : {}".format(wav_name))
!ls /kaggle/input/dc2020task2/
! wget https://raw.githubusercontent.com/daisukelab/dcase2020_task2_variants/master/file_info.csv



df = pd.read_csv('file_info.csv')

df.file = df.file.map(lambda f: str(f).replace('/data/task2/dev', '/kaggle/input/dc2020task2'))

types = df.type.unique()



df.head()
agg = df[['file', 'type', 'split']].groupby(['type', 'split']).agg('count')

fig = plt.figure(figsize=(12.0, 6.0))

g = sns.barplot(x="type", y="file", hue="split", data=agg.reset_index())

plt.ylabel("machine type")

plt.ylabel("number of files")

plt.show()

agg.transpose()
df.groupby(['type', 'split']).describe()
def get_wave_detail(filename):

    wav, sampling_rate = file_load(filename)



    n_fft = sampling_rate

    half = len(wav) // 2

    middle = wav[half - n_fft//2: half + n_fft//2]

    freq, P = signal.welch(middle, sampling_rate)



    return wav.shape, sampling_rate, wav.shape[-1]/sampling_rate, wav, P, freq



for t in types:

    for split in ['train', 'test']:

        type_df = df[df['type'] == t][df.split == split].reset_index()

        R = 4

        fig, ax = plt.subplots(R, 4, figsize = (15, 5*R//2))

        print(f'=== Machine type [{t}], {split} set ===')

        for i in range(R * 4):

            file_index = i % 4 + ((i // 8) * 4)

            file_path = Path(type_df.file[file_index])

            shape, sr, sec, wav, P, freq = get_wave_detail(file_path)

            assert int(sr) == 16000, f'{type(sr)} {sr}'

            if (i % 8) < 4:

                ax[i//4, i%4].set_title(file_path.name)

                ax[i//4, i%4].plot(freq, P)

                ax[i//4, i%4].set_xscale('log')

                ax[i//4, i%4].set_yscale('log')

            else:

                ax[i//4, i%4].plot(wav)

                ax[i//4, i%4].get_xaxis().set_ticks([])

        plt.show()
import sys



def get_log_mel_spectrogram(filename, n_mels=64,

                        n_fft=1024,

                        hop_length=512,

                        power=2.0):

    wav, sampling_rate = file_load(filename)

    mel_spectrogram = librosa.feature.melspectrogram(y=wav,

                                                     sr=sampling_rate,

                                                     n_fft=n_fft,

                                                     hop_length=hop_length,

                                                     n_mels=n_mels,

                                                     power=power)

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    return log_mel_spectrogram



for t in types:

    for split in ['train', 'test']:

        type_df = df[df['type'] == t][df.split == split].reset_index()

        R = 2

        fig, ax = plt.subplots(R, 1, figsize = (15, 2.5*R))

        print(f'=== Machine type [{t}], {split} set ===')

        for i in range(R * 1):

            file_index = i

            file_path = Path(type_df.file[file_index])

            mels = get_log_mel_spectrogram(file_path)

            ax[i].set_title(file_path.name)

            ax[i].imshow(mels)

            ax[i].axis('off')

        plt.show()