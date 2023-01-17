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



from scipy.io import wavfile



from sklearn.preprocessing import StandardScaler



from tensorflow.keras.preprocessing.sequence import pad_sequences



from tqdm import tqdm



from tensorflow.keras.utils import to_categorical



# Any results you write to the current directory are saved as output.
os.listdir('../input/audio-data')
train = pd.read_csv('../input/freesound-audio-tagging/train.csv')

train
audio_path = '../input/freesound-audio-tagging/audio_train/'

fname, label, verified = train.values[0]

rate, data = wavfile.read(audio_path+fname)



print(label)

print('Sampling Rate:\t{}'.format(rate))

print('Total Frames:\t{}'.format(data.shape[0]))

print(data)



y, sr = librosa.load(audio_path+fname,sr=11025)

IPython.display.Audio(data=y, rate=sr)

pad_audio_data = pad_sequences(np.load('../input/audio-data/audio_data_11025.npy', allow_pickle=True), maxlen=sr*5, value = 0, dtype = 'float32' )

pad_audio_data.shape
sds = StandardScaler()

sds_audio_data = sds.fit_transform(pad_audio_data)
labelEncoder = {}

for i, label in enumerate(train['label'].unique()):

    labelEncoder[label] = i

    
labelEncoder
Encoding_label = np.zeros(9473, dtype = object)



for i in tqdm(range(0,9473)):

    fname, label, verified = train.values[i]

    Encoding_label[i] = labelEncoder[label]

Encoding_label = to_categorical(Encoding_label,41)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D,Dense,Dropout,MaxPool1D,Flatten,GlobalMaxPool1D

from tensorflow.keras import optimizers, regularizers



input_length = sr * 5

n_classes = train['label'].unique().shape[0]

input_audio_data = np.expand_dims(sds_audio_data, axis=2)



sgd = optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)

momentum = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)

nag = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)

adagrad = optimizers.Adagrad(learning_rate=0.001)

rmsprop = optimizers.RMSprop(learning_rate=0.001, rho=0.9)

adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

#radam = RAdam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)

optimizer_dict = {sgd:'sgd', momentum:'momentum', nag:'nag', adagrad:'adagrad', rmsprop:'rmsprop', adam:'adam'}
def create_cnn():

    model = Sequential()

    model.add(Conv1D(16, 9, activation='relu', input_shape=(input_length, 1)))

    model.add(Conv1D(32, 9, activation='relu'))

    model.add(MaxPool1D(16))

    model.add(Dropout(0.1))

    model.add(Conv1D(64, 9, activation='relu'))

    model.add(Conv1D(64, 9, activation='relu'))

    model.add(MaxPool1D(16))

    model.add(Conv1D(128, 9, activation='relu'))

    model.add(Conv1D(128, 9, activation='relu'))

    model.add(MaxPool1D(4))

    model.add(Dropout(0.1))

    model.add(Conv1D(256, 3, activation='relu'))

    model.add(Conv1D(256, 3, activation='relu'))

    model.add(GlobalMaxPool1D())

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(1028, activation = 'relu'))

    model.add(Dense(n_classes, activation = 'softmax'))

    

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = adam)

    return model
model = create_cnn()

model.summary()
history = model.fit(input_audio_data,Encoding_label, epochs=30, validation_split = 1/6)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train','Validation'], loc = 'upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train','Validation'], loc = 'upper left')

plt.show()
import random



stop = train.shape[0]

rand = random.randrange(0, stop)

rand
fname, label, verified = train.values[rand]



y_predict, sr_predict = librosa.load(audio_path+fname,sr=11025)

IPython.display.Audio(data=y_predict, rate=sr_predict)
np.argmax(model.predict(np.expand_dims(input_audio_data[rand], axis=0)))
label
labelEncoder