# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master"))



# Any results you write to the current directory are saved as output.
import librosa

import os

from os.path import isdir, join

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import signal

from scipy.io import wavfile

from tensorflow.keras.preprocessing.sequence import pad_sequences
data_path = '../input/free-spoken-digit-dataset-master/free-spoken-digit-dataset-master/recordings'
def load_speeches(path):

    waves = [f for f in os.listdir(path) if f.endswith('.wav')]

    labels = []

    samples_rate = []

    all_waves = []

    for wav in waves:

        sample_rate, samples = wavfile.read(join(path,wav))

        samples_rate.append(sample_rate)

        labels.append(wav[0])

        all_waves.append(samples)

    return all_waves ,samples_rate,labels
def get_spectrograms(waves):

    sample_rate = 8000

    spectros = []

    freqs = []

    tims = []

    for wav in waves:

        frequencies, times, spectrogram = signal.spectrogram(wav, sample_rate)

        freqs.append(frequencies)

        tims.append(times)

        spectros.append(spectrogram)

    return freqs,tims,spectros

        
all_waves,samples_rate,labels = load_speeches(data_path)
max_sequence_len = max([len(x) for x in all_waves])

all_waves = np.array(pad_sequences(all_waves, maxlen=max_sequence_len, padding='post'))

freqs,tims,spectros = get_spectrograms(all_waves)
spectros[3].shape
spectros = np.array(spectros)

spectros = spectros.reshape(2000,129,81,1)
sns.countplot(labels)
import keras

labels = keras.utils.to_categorical(labels, 10)
from sklearn.model_selection import train_test_split

X, X_test, Y, Y_test = train_test_split(spectros, labels, test_size=0.2, random_state=42)
from tensorflow.keras.layers import Dense, Dropout, Flatten

from tensorflow.keras import Input, layers

from tensorflow.keras import backend as K

import tensorflow as tf
X.shape[1:]
model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(32, (5,5), activation='relu',padding='same', input_shape=(129, 81,1)),

  tf.keras.layers.Conv2D(32,(5,5), activation='relu',padding='same'),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Dropout((0.25)),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)),

  tf.keras.layers.Dropout((0.25)),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dropout((0.5)),

  tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

#model.compile(optimizer = tf.keras.optimizers.Adam( epsilon=1e-08), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X,Y,batch_size=512,epochs=100,validation_data=(X_test,Y_test))


%matplotlib inline

#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

history = model.history

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.figure(figsize=(10,7))

plt.plot(epochs, acc, 'r')

plt.plot(epochs, val_acc, 'b')

plt.title('Training and validation accuracy')

plt.legend(['Training accuracy','Validation accuracy'])

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.figure(figsize=(10,7))

plt.plot(epochs, loss, 'r')

plt.plot(epochs, val_loss, 'b')

plt.title('Training and validation loss')

plt.legend(['Training loss','Validation loss'])

plt.figure()

max(val_acc) #the best validation accuracy the model have got