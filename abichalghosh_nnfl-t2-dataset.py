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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

%matplotlib inline

import matplotlib.pyplot as plt

import os

from scipy import signal

import numpy as np

import librosa

import random as rn
DATA_DIR='/kaggle/input/number/Dataset/'
wav, sr = librosa.load('/kaggle/input/number/Dataset/1_Prateek_v3.wav')

print(sr,wav.shape)

plt.plot(wav)
from scipy.io.wavfile import read as read_wav

sampling_rate, data=read_wav("/kaggle/input/number/Dataset/4_satyansh_v1.wav")

print(sampling_rate)
import librosa.display

D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)

librosa.display.specshow(D, y_axis='linear')
mfcc = librosa.feature.mfcc(wav)

#padded_mfcc = pad2d(mfcc,40)

D = librosa.amplitude_to_db(mfcc, ref=np.max)

librosa.display.specshow(D)
librosa.get_duration(y=wav,sr=sr)
from keras.utils import to_categorical

import os



test_speaker = 'Ninad'

train_X = []

train_y = []



test_X = []

test_y = []





for fname in os.listdir(DATA_DIR):

    try:

        if '.wav' not in fname or 'dima' in fname:

            continue

        struct = fname.split('_')

        if(len(struct[0])<=1 or struct[0]=='10'):

            digit = struct[0]

        elif(struct[0]=='10h'):

            digit = '21'

        else:

            temp=int(struct[0][0])

            temp+=11

            digit=str(temp)

#         digit = struct[0]

        print(digit)

        speaker = struct[1]

        wav, sr = librosa.load(DATA_DIR + fname)

        if speaker == test_speaker:

            test_X.append(wav)

            test_y.append(digit)

        else:

            train_X.append(wav)

            train_y.append(digit)

    except Exception as e:

        print(fname, e)

        raise

print('train_X:', len(train_X))

print('train_y:', len(train_y))

print('test_X:', len(test_X))

print('test_y:', len(test_y))
print(train_y[4])

print(type(train_X[4]))

plt.plot(train_X[4])
print(librosa.__version__)
import soundfile as sf

sf.write('/kaggle/working/output1.wav', data=train_X[3], samplerate=48000, subtype=None, endian=None, format=None, closefd=True)

# sampling_rate, data=read_wav("/kaggle/working/output1.wav")

# print(sampling_rate)
import IPython.display as ipd

ipd.Audio('/kaggle/working/output1.wav')