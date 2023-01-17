import os

import numpy as np

import pandas as pd

import seaborn as sns

import librosa

import librosa.display



import wave

from matplotlib import pyplot as plt

from IPython import display as ipd

from scipy.io import wavfile
import warnings

warnings.filterwarnings('ignore')
# Script containing lwlrap functions

%run ../usr/lib/2019_fat_lwlrap/2019_fat_lwlrap.py
SAMPLE_RATE = 44100

N_MFCC = 50

FIGSIZE = (16,9)

WIDTH = 1.0
DATA_PATH = '../input/freesound-audio-tagging-2019/'



CURATED_CSV = DATA_PATH + 'train_curated.csv'

CURATED_DIR = DATA_PATH + 'train_curated/'

NOISY_CSV = DATA_PATH + 'train_noisy.csv'

NOISY_DIR = DATA_PATH + 'train_noisy/'

SAMPLE_CSV = DATA_PATH + 'sample_submission.csv'
def gen_duration(path):

    def f(file):

        with wave.open(path+file) as data:

            return data.getnframes() / SAMPLE_RATE

    return f



train_curated = pd.read_csv(CURATED_CSV)

train_curated['curated'] = True

train_curated['duration'] = train_curated.fname.apply(gen_duration(CURATED_DIR))



train_noisy = pd.DataFrame()

train_noisy = pd.read_csv(NOISY_CSV)

train_noisy['curated'] = False

train_noisy['duration'] = train_noisy.fname.apply(gen_duration(NOISY_DIR))



train = pd.concat([train_curated, train_noisy], ignore_index=True)
# N-hot for labels

curated_labels = train[train.curated==True].labels.str.get_dummies(sep=',')

noisy_labels = train[train.curated==False].labels.str.get_dummies(sep=',')
# plt.figure(figsize=FIGSIZE)



plt.subplot(121)

ax = curated_labels.sum(axis=1).value_counts().sort_index().plot.bar()

ax.set_title('File with N labels : Curated')

ax.set_xlabel('label count')

ax.set_ylabel('count')

plt.tight_layout()



plt.subplot(122)

ax = noisy_labels.sum(axis=1).value_counts().sort_index().plot.bar()

ax.set_title('File with N labels : Noisy')

ax.set_xlabel('label count')

ax.set_ylabel('count')

plt.tight_layout()
plt.figure(figsize=FIGSIZE)



ax = curated_labels.sum().plot.bar(color='r', width=WIDTH)

ax = curated_labels[curated_labels.sum(axis=1)==1].sum().plot.bar(color='b', width=WIDTH)

ax.set_title('samples per label : Curated');

plt.tight_layout()
plt.figure(figsize=FIGSIZE)



ax = noisy_labels.sum().plot.bar(color='r', width=WIDTH)

ax = noisy_labels[noisy_labels.sum(axis=1)==1].sum().plot.bar(color='b', width=WIDTH)

ax.set_title('samples per label : Noisy')

plt.tight_layout();
plt.figure(figsize=FIGSIZE)



labels = train.labels.str.get_dummies(sep=',')

ax = labels.sum().plot.bar(color='r', width=WIDTH)

ax = labels[labels.sum(axis=1)==1].sum().plot.bar(color='b', width=WIDTH)

ax.set_title('samples per label : Combined')

plt.tight_layout();
plt.figure(figsize=FIGSIZE)



N_BIN = 50



# upper half

plt.subplot(211)

ax1 = train.duration.hist(bins=N_BIN, color='r')

ax1 = train[train.curated==True].duration.hist(bins=N_BIN)

plt.ylim([17000,18500])



# lower half

plt.subplot(212)

ax2 = train.duration.hist(bins=N_BIN, color='r')

ax2 = train[train.curated==True].duration.hist(bins=N_BIN)

ax2.set_xlabel('Duration (sec)')

ax2.set_ylabel('count');

plt.ylim([0,1500]);





# hide the spines between ax and ax2

ax1.spines['bottom'].set_visible(False)

ax2.spines['top'].set_visible(False)

ax1.xaxis.tick_top()

ax1.tick_params(labeltop='off')  # don't put tick labels at the top

ax2.xaxis.tick_bottom()



d = .01  # how big to make the diagonal lines in axes coordinates

# arguments to pass to plot, just so we don't keep repeating them

kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)

ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal



kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes

ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs); # bottom-right diagonal
durations = []

labels = []

for index, row in train[(train.curated==True)].iterrows():

    arr = row.labels.split(',')

    durations.extend([row.duration]*len(arr))

    labels.extend(arr)



plt.figure(figsize=FIGSIZE)

plt.grid()

ax = sns.boxplot(x=np.array(labels), y=np.array(durations));

ax.set_title('Duration distribution per class')

plt.xticks(rotation=90);
def audio_analysis(x, draw_plot=True):

    

    if type(x) is str:

        rows = train[train.fname==x]

        assert len(rows)>0, f'"{x}" not found in training dataframe.'

        x = rows.iloc[0]

    

    

    DIR = CURATED_DIR if x.curated else NOISY_DIR

    

    # Obtain audio data

    rate, data = wavfile.read(DIR + x.fname)

    data = data.astype(float)

    data /= data.max()

    

    if draw_plot:

        # Plot

        plt.figure(figsize=FIGSIZE)



        # Waveform

        ax = plt.subplot(311)

        ax.set_title('{} / {} / {}'.format(x.fname, x.labels, f'Curated={x.curated}'))

        ax.set_ylabel('Amplitude')    

        librosa.display.waveplot(data, sr=rate)



        # Spectrogram

        ax = plt.subplot(312)

        S = librosa.feature.melspectrogram(data, sr=rate, n_mels=128)

        log_S = librosa.power_to_db(S, ref=np.max)

        librosa.display.specshow(log_S, sr=rate, x_axis='time', y_axis='mel')

        ax.set_title('Mel power spectrogram ')

        plt.colorbar(format='%+02.0f dB')

        plt.tight_layout()



        # MFCC coefficient

        ax = plt.subplot(313)

        mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        librosa.display.specshow(delta2_mfcc)

        plt.ylabel('MFCC coeffs')

        plt.xlabel('Time')

        plt.title('MFCC')

        plt.colorbar()

        plt.tight_layout()

    

    return ipd.Audio(filename=DIR+x.fname)
# numerous labels

audio_analysis('4dad3998.wav')
# Distinct labels

audio_analysis('d828ffce.wav')
# Mixed with other sounds

audio_analysis('8250b48f.wav')
# Incorrect label '6a1f682a.wav'

audio_analysis('f76181c4.wav')