# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

from tqdm import tqdm

import pickle

import IPython.display as ipd  # To play sound in the notebook
# Use one audio file in previous parts again

fname = '/kaggle/input/indian-music-raga/bhoopali22.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
def noise(data):

    """

    Adding White Noise.

    """

    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html

    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5

    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])

    return data
x = noise(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
def shift(data):

    """

    Random Shifting.

    """

    s_range = int(np.random.uniform(low=-5, high = 5)*1000)  #default at 500

    return np.roll(data, s_range)
x = shift(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
def stretch(data, rate=0.8):

    """

    Streching the Sound. Note that this expands the dataset slightly

    """

    data = librosa.effects.time_stretch(data, rate)

    return data
x = stretch(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
def pitch(data, sample_rate):

    """

    Pitch Tuning.

    """

    bins_per_octave = 12

    pitch_pm = 2

    pitch_change =  pitch_pm * 2*(np.random.uniform())   

    data = librosa.effects.pitch_shift(data.astype('float64'), 

                                      sample_rate, n_steps=pitch_change, 

                                      bins_per_octave=bins_per_octave)

    return data
x = pitch(data, sampling_rate)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
def dyn_change(data):

    """

    Random Value Change.

    """

    dyn_change = np.random.uniform(low=-0.5 ,high=7)  # default low = 1.5, high = 3

    return (data * dyn_change)
x = dyn_change(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)
def speedNpitch(data):

    """

    peed and Pitch Tuning.

    """

    # you can change low and high here

    length_change = np.random.uniform(low=0.8, high = 1)

    speed_fac = 1.2  / length_change # try changing 1.0 to 2.0 ... =D

    tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)

    minlen = min(data.shape[0], tmp.shape[0])

    data *= 0

    data[0:minlen] = tmp[0:minlen]

    return data
x = speedNpitch(data)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(x, sr=sampling_rate)

ipd.Audio(x, rate=sampling_rate)