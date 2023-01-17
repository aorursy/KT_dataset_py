# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import librosa

import librosa.display

import IPython.display

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.
os.listdir('../input/train-test')
train = pd.read_csv('../input/freesound-audio-tagging/train.csv')

train
audio_path = '../input/freesound-audio-tagging/audio_train/'

from scipy.io import wavfile

fname, label, verified = train.values[0]

rate, data = wavfile.read(audio_path+fname)



print(label)

print('Sampling Rate:\t{}'.format(rate))

print('Total Frames:\t{}'.format(data.shape[0]))

print(data)



y, sr = librosa.load(audio_path+fname)

IPython.display.Audio(data=y, rate=sr)

#a = np.load('../input/train-test/train_test.npy', allow_pickle=True)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_audio_data = pad_sequences(np.load('../input/train-test/train_test.npy', allow_pickle=True), maxlen=sr*2, value = 0, dtype = 'float32' )

pad_audio_data.shape
labelEncoder = {}

for i, label in enumerate(train['label'].unique()):

    labelEncoder[label] = i

    
labelEncoder
from tqdm import tqdm
Encoding_label = np.zeros(9473, dtype = object)



for i in tqdm(range(0,9473)):

    fname, label, verified = train.values[i]

    Encoding_label[i] = labelEncoder[label]

from tensorflow.keras.utils import to_categorical
Encoding_label = to_categorical(Encoding_label,41)
plt.plot(data[:1024])
D = librosa.amplitude_to_db(librosa.stft(y[:1024]),ref=np.max)



plt.plot(D.flatten())

plt.show()
S = librosa.feature.melspectrogram(y, sr=sr)



plt.figure(figsize=(12,4))

librosa.display.specshow(librosa.power_to_db(S,ref=np.max), sr=sr, x_axis='time', y_axis='mel')

plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.show()
mfcc = librosa.feature.mfcc(y=y, sr=sr)





plt.figure(figsize=(12,4))

librosa.display.specshow(mfcc, x_axis='time')

plt.colorbar(format='%+2.0f dB')

plt.tight_layout()

plt.show()
min_level_db = -100

 

def _normalize(S):

    return np.clip((librosa.power_to_db(S,ref=np.max) - min_level_db) / -min_level_db, 0, 1)

norm_S = _normalize(S)



plt.figure(figsize=(12, 4))

librosa.display.specshow(norm_S, sr=sr, x_axis='time', y_axis='mel')

plt.title('norm mel power spectrogram')

plt.colorbar(format='%+0.1f dB')

plt.tight_layout()

plt.show()
from keras.models import Sequential

from keras.layers import Conv1D,Dense,Dropout,MaxPool1D,Flatten

from keras import optimizers



input_length = sr * 2 

n_classes = train['label'].unique().shape[0]

input_audio_data = np.expand_dims(pad_audio_data, axis=2)

opt = optimizers.Adam(learning_rate=0.00001)





def create_cnn():

    model = Sequential()

    model.add(Conv1D(filters=4, kernel_size=16, activation='relu', padding='same', input_shape=(input_length, 1)))

    model.add(MaxPool1D(pool_size=2))

    model.add(Dropout(rate=0.1))

    model.add(Conv1D(filters=6, kernel_size=16, activation='relu', padding='same'))

    model.add(MaxPool1D(pool_size=2))

    model.add(Dropout(rate=0.1))

    model.add(Conv1D(filters=9, kernel_size=16, activation='relu', padding='same'))

    model.add(MaxPool1D(pool_size=2))

    model.add(Dropout(rate=0.1))

    model.add(Flatten())

    model.add(Dense(units=100, activation = 'relu'))

    model.add(Dense(units=n_classes, activation = 'softmax'))

    

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = opt)

    return model
model = create_cnn()

model.summary()
history = model.fit(input_audio_data,Encoding_label, epochs=20, validation_split = 1/6)