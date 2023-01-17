# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Imports
import pickle
import keras
import random
import scipy

from pathlib import Path
from subprocess import check_output

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

%matplotlib inline
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio'
with open('../input/speech-recognition-data-processing/SavedPartition.pickle', 'rb') as handle:
    partition = pickle.load(handle)
with open('../input/speech-recognition-data-processing/SavedLabels.pickle', 'rb') as handle:
    labels = pickle.load(handle)
print("loaded data.")
# Data Augmenting
def bandpass(sample_rate, samples):
    
    fs = sample_rate  # Sample frequency (Hz)
    fl = 180.0  # Human voices range from 85 Hz to 255 Hz
    fh = 240.0
    Q = 1.0  # Quality factor
    w0 = fl/(fs/2)  # Normalized Frequency
    w1 = fh/(fs/2)
    # Design notch filter
    b, a = scipy.signal.butter(3, [w0, w1], btype='bandpass', analog=True)
    samples = scipy.signal.lfilter(b,a,samples)*30

    return sample_rate, samples
def spectrogram(sample_rate, samples):
    eps=1e-10
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    # silence can end up being empty files, in this case we can just return one second of zeros
    if len(spectrogram.shape) < 2:
        return np.zeros((71,129))
    else:
        return np.log(np.abs(spectrogram).T+eps)

def stft(sample_rate, samples):

    eps=1e-10

    frequencies, times, Zxx = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    
    # silence can end up being empty files, in this case we can just return one second of zeros
    if len(Zxx.shape) < 2:
        return np.zeros((151,161))
    else:
        return np.log(np.abs(Zxx).T+eps)
class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, batch_size=32, dim=(151,161), x1dim=(71,129), x2dim=(151,161), n_channels=1,
                 n_classes=11, shuffle=True, input_type='wav', testing=False, augment=True):
        # Initialization
        self.dim = dim
        self.x1dim = x1dim
        self.x2dim = x2dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.input_type = input_type
        self.testing = testing
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.input_type == 'all':
            X1, X2, y = self.__data_generation(list_IDs_temp)
            if self.testing:
                return [X1,X2]
            else:
                return [X1,X2], [y]
        else:
            X, y = self.__data_generation(list_IDs_temp)
            
            return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        # Initialization

        if self.input_type == 'all':
            X1 = np.empty((self.batch_size, *self.x1dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.x2dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data  
        if self.input_type == 'spectrogram':
            for i, ID in enumerate(list_IDs_temp):
                if self.labels[ID] == 11:
                    path = '../input/'
                else:
                    path = train_audio_path + '/'
                sample_rate, samples = wavfile.read(path + ID)

                if self.augment:
                    if random.randint(1,101) < 51:
                        sample_rate, samples = bandpass(sample_rate, samples)
                        
                trans = spectrogram(sample_rate, samples)
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.x1dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
                
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        elif self.input_type == 'stft':
            for i, ID in enumerate(list_IDs_temp):
                if self.labels[ID] == 11:
                    path = '../input/'
                else:
                    path = train_audio_path + '/'
                sample_rate, samples = wavfile.read(path + ID)

                if self.augment:
                    if random.randint(1,101) < 51:
                        sample_rate, samples = bandpass(sample_rate, samples)
                        
                trans = stft(sample_rate, samples)
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.x2dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
            
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        elif self.input_type == 'all':
        
            for i, ID in enumerate(list_IDs_temp):
                if self.labels[ID] == 11:
                    path = '../input/'
                else:
                    path = train_audio_path + '/'
                sample_rate, samples = wavfile.read(path + ID)

                if self.augment:
                    if random.randint(1,101) < 51:
                        sample_rate, samples = bandpass(sample_rate, samples)
                        
                trans = spectrogram(sample_rate, samples)
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.x1dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X1[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
            for i, ID in enumerate(list_IDs_temp):
                if self.labels[ID] == 11:
                    path = '../input/'
                else:
                    path = train_audio_path + '/'
                sample_rate, samples = wavfile.read(path + ID)

                if self.augment:
                    if random.randint(1,101) < 51:
                        sample_rate, samples = bandpass(sample_rate, samples)
                        
                trans = stft(sample_rate, samples)
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.x2dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X2[i,] = padded[:, :, np.newaxis]
                
            return X1, X2, keras.utils.to_categorical(y, num_classes=self.n_classes)
from keras.models import load_model

spect_model = load_model('../input/spect-model/spect_model.h5')
stft_model = load_model('../input/stft-model/stft_model.h5')
spect_model.trainable = False
spect_model.layers.pop()
spect_model.compile

for layer in spect_model.layers:
    layer.name = "spect_" + layer.name
    
stft_model.trainable = False
stft_model.layers.pop()
stft_model.compile

for layer in stft_model.layers:
    layer.name = "stft_" + layer.name
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import concatenate, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# Parameters
params = {'x1dim': (71,129),
          'x2dim': (151,161),
          'batch_size': 64,
          'n_classes':12,
          'n_channels': 1,
          'shuffle': True,
          'input_type': 'all',
          'augment': True
         }

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

spect_model_output = spect_model.get_layer('spect_dropout_3').output
stft_model_output = stft_model.get_layer('stft_dropout_3').output

concatenated = concatenate([spect_model_output, stft_model_output])
prediction = Dense(12, activation='softmax', name='prediction')(concatenated)
        
ensemble_model = Model([spect_model.input, stft_model.input], prediction)

for layer in ensemble_model.layers:
    if not(layer.name) == 'prediction':
        layer.trainable = False

ensemble_model.compile(optimizer = 'adam',
                       loss='categorical_crossentropy',
                       metrics=["accuracy"])

ensemble_model.summary()

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# Train model on dataset
history = ensemble_model.fit_generator(generator=training_generator,
                                       validation_data=validation_generator,
                                       steps_per_epoch=30,
                                       epochs=20,
                                       verbose=2,
                                       callbacks=[annealer]
                                      )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('CNN Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('CNN Loss')
plt.legend()

plt.show()
ensemble_model.save('ensemble_model.h5')