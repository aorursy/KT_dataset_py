# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import librosa

import IPython.display as ipd

import matplotlib.pyplot as plt

import numpy as np

from scipy.io import wavfile

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D

from keras.models import Model

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras.models import load_model

import random

# import sounddevice as sd

import soundfile as sf

from keras.utils import to_categorical

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
os.listdir('../input/tensorflow-speech-recognition-challenge/')
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio/'

samples, sample_rate = librosa.load(train_audio_path+'yes/0a7c2a8d_nohash_0.wav', sr = 16000)

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(211)

ax1.set_title('Raw wave of ' + '../input/train/audio/yes/0a7c2a8d_nohash_0.wav')

ax1.set_xlabel('time')

ax1.set_ylabel('Amplitude')

ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)