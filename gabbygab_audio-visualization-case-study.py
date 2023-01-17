%%capture

!apt-get install libav-tools -y



%matplotlib inline

import os

import pandas as pd

import numpy as np

import librosa

import librosa.display

import seaborn as sns

import torchvision.models as models

import matplotlib

import matplotlib.pyplot as plt

import IPython.display as ipd





sns.set(style='whitegrid')

plt.style.use('seaborn-darkgrid')

#seaborn-ticks

from fastai.callbacks import *

from sklearn.metrics import roc_curve, auc

from fastai.vision import *

from glob import glob
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print('Train Size: ', df_train.shape)

print('Test Size: ', df_test.shape)
Class = df_train.Class.value_counts()

colours = ["#aaaaaa", "#aaaaaa", "#aaaaaa","#aaaaaa","#aaaaaa","#d11111","#aaaaaa","#aaaaaa","#aaaaaa","#d11111"]

f, ax = plt.subplots(figsize=(18,5)) 

ax = sns.countplot(x='Class', data=df_train, palette=colours)

plt.title('Class Distribution');
i = random.choice(df_train.index)

y, sr = librosa.load('../input/train/Train/' + str(df_train.ID[i]) + '.wav')

plt.figure(figsize=(12,5))

librosa.display.waveplot(y,sr);

print('Class: ', df_train.Class[i])

print('Sampling Rate: ',sr,'Hz')

print('Duration: ',len(y)/sr)

print('Number of samples: ', len(y))

ipd.Audio(data=y, rate=sr)
y, sr = librosa.load('../input/train/Train/68.wav')

plt.figure(figsize=(12,5))

librosa.display.waveplot(y,sr);

plt.title('Dog_bark')

ipd.Audio(data=y, rate=sr)
plt.figure(figsize=(12,5))

noise = np.random.randn(len(y))

data_noise = y + 0.05 * noise

librosa.display.waveplot(data_noise,sr);

plt.title('Dog_bark with background noise')

ipd.Audio(data=data_noise, rate=sr)
plt.figure(figsize=(12,5))

y_fast = librosa.effects.time_stretch(y, 0.5)

librosa.display.waveplot(y_fast,sr);

plt.title('Dog_bark')

ipd.Audio(data=y_fast, rate=sr)
plt.figure(figsize=(12,5))

y_third = librosa.effects.pitch_shift(y, sr, n_steps=20)

librosa.display.waveplot(y_third,sr)

plt.title('Dog_bark')

ipd.Audio(data=y_third, rate=sr)
# Loading audio

def load_sound_files(parent_dir, file_paths):

    raw_sounds = []

    for fp in file_paths:

        X,sr = librosa.load(parent_dir + str(fp))

        raw_sounds.append(X)

    return raw_sounds



#plot waveplots

def plot_waves(sound_names,raw_sounds):

    i = 1

    fig = plt.figure(figsize=(25,12), dpi = 900)

    for n,f in zip(sound_names,raw_sounds):

        plt.subplot(2,5,i)

        librosa.display.waveplot(np.array(f), sr)

        plt.title(n.title())

        i += 1

    plt.suptitle('Waveplot',x=0.5, y=0.95,fontsize=18)

    plt.show()



#plot fourier transform

def plot_fft(sound_names,raw_sounds):

    i = 1

    fig = plt.figure(figsize=(25,12), dpi = 900)

    for n,f in zip(sound_names,raw_sounds):

        plt.subplot(2,5,i)

        X = scipy.fft(f)

        X_mag = np.absolute(X)

        f = np.linspace(0, sr, len(X_mag))

        plt.title(n.title())

        plt.plot(f, X_mag)

        plt.xlabel('Frequency')

        i += 1

    plt.suptitle('Fourier Transform',x=0.5, y=0.95,fontsize=18)

    plt.show()

    

    

#plot Mel-scale-power Spectrogram

def plot_mel_specgram(sound_names,raw_sounds):

    i = 1

    fig = plt.figure(figsize=(25,10), dpi = 900)

    for n,f in zip(sound_names,raw_sounds):

        plt.subplot(2,5,i)

        S = librosa.feature.melspectrogram(y=f, sr=sr,n_fft=2048, hop_length=512)

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel')

        plt.title(n.title())

        i += 1

    plt.suptitle('Mel-scaled power spectrogram',x=0.5, y=0.95,fontsize=18)

    plt.show()

    

#plot Mfccs

def plot_mfccs(sound_names,raw_sounds):

    i = 1

    fig = plt.figure(figsize=(25,10), dpi = 900)

    for n,f in zip(sound_names,raw_sounds):

        plt.subplot(2,5,i)

        S = librosa.feature.mfcc(y=f, sr=sr, dct_type=2)

        librosa.display.specshow(S)

        plt.title(n.title())

        i += 1

    plt.suptitle('Mel frequency cepstral coefficients ',x=0.5, y=0.95,fontsize=18)

    plt.show()

    

#plot Spectrogram Constrast    

def plot_chroma(sound_names,raw_sounds):

    i = 1

    fig = plt.figure(figsize=(25,10), dpi = 900)

    for n,f in zip(sound_names,raw_sounds):

        plt.subplot(2,5,i)

        S = np.abs(librosa.stft(f))

        chroma = librosa.feature.chroma_stft(S=S, sr=sr)

        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')

        plt.title(n.title())

        i += 1

    plt.suptitle('Chromagram of a short time fourier transform ',x=0.5, y=0.95,fontsize=18)

    plt.show()

    

#plot ROC_AUC

def cal_auc_and_plot(learn):

    preds, y = learn.get_preds()

    probs = np.exp(preds[:,1])

    fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([-0.01, 1.0])

    plt.ylim([0.0, 1.01])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic')

    plt.legend(loc="lower right")

    return roc_auc

    

sound_file_paths = ['77.wav','88.wav','6.wav','71.wav','2.wav','17.wav','59.wav','104.wav','3.wav','1.wav']

sound_names = list(np.unique(df_train.Class))

parent_dir = "../input/train/Train/"



raw_sounds = load_sound_files(parent_dir, sound_file_paths)

plot_waves(sound_names, raw_sounds)
plot_fft(sound_names,raw_sounds)
plot_mel_specgram(sound_names, raw_sounds)
plot_mfccs(sound_names, raw_sounds)
plot_chroma(sound_names, raw_sounds)