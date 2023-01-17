%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import scipy as sp

import scipy.stats as stats

import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import re

from pathlib import Path

import pickle as pkl

from time import time

from tqdm import tqdm, tnrange, tqdm_notebook

from pprint import pprint

import os, sys

from warnings import warn

import itertools

import librosa

import librosa.display
BASE = Path('../input')
# Load some data segments



def get_segment(skiprows=None, samples_per_segment = 150000):

    df = pd.read_csv(BASE/'train.csv', nrows=samples_per_segment, skiprows=skiprows,

                 dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    df.rename(columns={'acoustic_data': 'signal', 'time_to_failure': 'quaketime'}, inplace=True)

    return df





df = get_segment()

df2 = get_segment(range(1,int(2.2e6)))

df_peak = get_segment(range(1,int(4.4e6)))
def plot_spec(df, n_fft=4096, hop_length=1024, sr=4096e3, n_mels=64, fmin=1480, fmax=640e3):

    stft = librosa.stft(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length)

    stft_magnitude, stft_phase = librosa.magphase(stft)

    stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)



    fig,axs = plt.subplots(2,2, figsize=(12,8))

    librosa.display.specshow(stft_magnitude, ax=axs[0,0], x_axis='time', y_axis='linear', 

                             sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    axs[0,0].set_ylim((fmin,fmax))

    axs[0,0].set_title(f'Spectrogram ({stft_magnitude_db.shape}, hop={hop_length:d}, Linear)')



    librosa.display.specshow(stft_magnitude_db, ax=axs[0,1], x_axis='time', y_axis='linear', 

                             sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    axs[0,1].set_ylim((fmin,fmax))

    axs[0,1].set_title(f'Spectrogram ({stft_magnitude_db.shape}, hop={hop_length:d}, DB)')



    mel_spec = librosa.feature.melspectrogram(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length,

                                              n_mels=n_mels, sr=sr, power=1.0,

                                              fmin=fmin, fmax=fmax)

    librosa.display.specshow(mel_spec, ax=axs[1,0], x_axis='time', y_axis='mel', 

                             sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    axs[1,0].set_title(f'Spectrogram ({mel_spec.shape}, hop={hop_length:d}, Linear)')



    mel_spec = librosa.feature.melspectrogram(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length,

                                              n_mels=n_mels, sr=sr, power=1.0,

                                              fmin=fmin, fmax=fmax)

    mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

    librosa.display.specshow(mel_spec, ax=axs[1,1], x_axis='time', y_axis='mel', 

                             sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

    axs[1,1].set_title(f'Spectrogram ({mel_spec.shape}, hop={hop_length:d}, DB)')

    

    plt.tight_layout()



plot_spec(df)

plot_spec(df_peak)
def plot_spec(dfs, n_fft=4096, hop_length=None, sr=44e3, n_mels=256, fmin=0, fmax=128e3, power=2.0, lin=False):

    if hop_length is None:

        hop_length = n_fft // 4

    fig,axs = plt.subplots(1+lin,len(dfs), figsize=(6*len(dfs),4+4*lin))

    for i,df in enumerate(dfs):

        ax = axs[0,i] if lin else axs[i]

        mel_spec = librosa.feature.melspectrogram(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length,

                                                  n_mels=n_mels, sr=sr, power=power)#, fmin=fmin, fmax=fmax)

        mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

        l = mel_spec.shape[0]

        mel_spec = mel_spec[int(0.02*l):int(0.75*l),:]

        librosa.display.specshow(mel_spec, ax=ax, x_axis='time', y_axis='mel', 

                                 sr=sr, hop_length=hop_length)#, fmin=fmin, fmax=fmax)

        ax.set_title(f'Spectrogram ({mel_spec.shape}, hop={hop_length:d})')

        

        if lin:

            ax = axs[1,i]

            stft = librosa.stft(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length)

            stft_magnitude, stft_phase = librosa.magphase(stft)

            stft_magnitude_db = librosa.amplitude_to_db(stft_magnitude)

            librosa.display.specshow(stft_magnitude_db, ax=ax, x_axis='time', y_axis='linear', 

                                     sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax)

            ax.set_ylim((fmin,fmax))

            ax.set_title(f'Spectrogram ({stft_magnitude_db.shape}, hop={hop_length:d})')





plot_spec((df,df2,df_peak), sr=44e3, n_fft=4096, n_mels=256, power=2.0)
from scipy.misc import imread

seg_len = int(150e3)

n_fft=4096

hop_length=1024

sr=44e3

n_mels=256

power=2.0

class_limits=(0,1,2,3,4,5,6,7,8,10,12,99)
# Get 20 random segments + a peak segment, generate their spectrogram-arrays and PNG files.



np.random.seed(0)

sigs = []

specs = []

mels = []

ims = []

hops = list(np.random.randint(0,6000,20)) + [int(4.4e6/hop_length)]

for hop in hops:

    # load signal

    tmp = pd.read_csv(BASE/'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},

                      nrows=seg_len, skiprows=range(1,1+hop*hop_length))

    tmp.rename(columns={'acoustic_data': 'signal', 'time_to_failure': 'quaketime'}, inplace=True)

    # extract signal

    sigs.append(tmp.signal.values)

    # generate spec

    mel_spec = librosa.feature.melspectrogram(np.array(tmp.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length,

                                          n_mels=n_mels, sr=sr, power=power)

    specs.append(mel_spec)

    mel_spec = librosa.amplitude_to_db(mel_spec, ref=10**8)

    l = mel_spec.shape[0]

    mel_spec = mel_spec[int(0.02*l):int(0.75*l),:]

    mels.append(mel_spec)

    # save as PNG and load image

    fig = plt.figure(figsize=(2.6,2.6))

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    librosa.display.specshow(mel_spec, ax=plt.axes(), x_axis='time', y_axis='mel', vmin=-120, vmax=-20, # -50,50

                             sr=sr, hop_length=hop_length)

    plt.axes().set_axis_off()

    plt.savefig('tmp_spec.png')

    plt.close()

    ims.append(imread('tmp_spec.png'))
# Look for relation between the signal "power" (measured by mean(|signal|))

# and the PNG image color range (measured by sum(color^2) for each channel separately).



fig, axs = plt.subplots(1,3, figsize=(15,5))

for channel in range(3):

    axs[channel].plot([np.mean(np.abs(m.flatten())) for m in sigs],

                      [np.mean(([t**2 for t in im[:,:,channel].flatten()])) for im in ims],'.')

    axs[channel].set_xlabel('mean |signal|')

    axs[channel].set_ylabel('sum(color^2)')

    axs[channel].set_title(f'Channel {channel:d}')

    axs[channel].set_xscale('log')

    axs[channel].set_yscale('log')

    axs[channel].grid()

plt.tight_layout()
fig, axs = plt.subplots(7,3, figsize=(12,18))



for i,(s,im) in enumerate(zip(sigs,ims)):

    ax = axs[i%7,i//7]

    ax.imshow(im)

    ax.set_axis_off()

    ax.set_title(f'mean(|signal|) = {np.mean(np.abs(s.flatten())):.1f}')



plt.tight_layout()
BASE = Path('../input')

TRAIN = Path('../specs')



n_all = 629145480 # df_full.shape[0]

seg_len = int(150e3)

hop_len = int(25e3)

n_hops = int(n_all/hop_len)

segs_to_read = 1050

hops_to_read = int(segs_to_read * seg_len/hop_len)
def save_spec(df, seg_index, meta, class_limits=(0,1,2,3,4,5,6,7,8,10,12,99), base_path=TRAIN,

              n_fft=4096, hop_length=1024, sr=44e3, n_mels=256, power=2.0, fig=None):

    

    # get ttf and derive the filename

    tf = df.quaketime.values[-1]

    cls = np.where([a<=tf<b for a,b in zip(class_limits[:-1],class_limits[1:])])[0][0]

    cls_nm = '-'.join((f'{class_limits[cls]:02d}',f'{class_limits[cls+1]:02d}'))

    fname = str(seg_index) + '_' + cls_nm + '.png'

    

    # compute spectrogram

    mel_spec = librosa.feature.melspectrogram(np.array(df.signal,dtype='float'), n_fft=n_fft, hop_length=hop_length,

                                              n_mels=n_mels, sr=sr, power=power)

    mel_spec = librosa.amplitude_to_db(mel_spec, ref=10**8)

    l = mel_spec.shape[0]

    mel_spec = mel_spec[int(0.02*l):int(0.75*l),:]

    meta.loc[seg_index] = [fname, tf, cls_nm]

    

    # save as PNG

    if fig is None:

        fig = plt.figure(figsize=(2.6,2.6))

    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    librosa.display.specshow(mel_spec, ax=plt.axes(), x_axis='time', y_axis='mel', vmin=-50, vmax=50,

                             sr=sr, hop_length=hop_length)

    plt.axes().set_axis_off()

    plt.savefig(base_path/fname)

    plt.clf()
GENERATE_SPECTS = False



if GENERATE_SPECTS:

    meta = pd.DataFrame(columns=('filename','time','class'))

    fig = plt.figure(figsize=(2.6,2.6))

    for i in tqdm_notebook(range(n_hops)):

        if i % hops_to_read == 0:

            tmp = pd.read_csv(BASE/'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32},

                              nrows=seg_len*(segs_to_read+1), skiprows=range(1,1+i*hop_len))

            tmp.rename(columns={'acoustic_data': 'signal', 'time_to_failure': 'quaketime'}, inplace=True)

        save_spec(tmp[(i%hops_to_read)*hop_len:(i%hops_to_read)*hop_len+seg_len], i, meta, fig=fig)



    meta.to_csv(TRAIN/'train_spec_meta.csv', index=False)

    

    plt.figure(figsize=(10,6))

    plt.plot(meta.time)

    plt.xlabel('Training segment')

    plt.ylabel('TTF')

    meta.head()