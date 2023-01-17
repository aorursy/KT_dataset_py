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
#Imports
import keras
import pickle
import random
import scipy

from keras.models import load_model

from pathlib import Path
from subprocess import check_output

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
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
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio'

def wavread(file, label):
    if label == 11:
        path = '../input/'
    else:
        path = train_audio_path + '/'
        
    sample_rate, samples = wavfile.read(path + file)
    return np.array(samples)

def spectrogram(sample_rate, samples):
    eps=1e-10
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    # silence can end up being empty files, in this case we can just return one second of zeros
    if len(spectrogram.shape) < 2:
        return np.zeros((71,129))
    else:
        return np.log(np.abs(spectrogram).T+eps)
class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, batch_size=32, dim=(151,161), x1dim=16000, x2dim=(151,161), n_channels=1,
                 n_classes=12, shuffle=True, input_type='wav', testing=False, augment=True):
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
         # move wavfile.read into datagenerator...
        
            
        if self.input_type == 'wav':
            X = np.empty((self.batch_size, self.dim, self.n_channels))
        elif self.input_type == 'all':
            X1 = np.empty((self.batch_size, self.x1dim, self.n_channels))
            X2 = np.empty((self.batch_size, *self.x2dim, self.n_channels))
        else:
            X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        
        if self.input_type == 'wav':
            for i, ID in enumerate(list_IDs_temp):
                wav = wavread(ID, self.labels[ID])
                padded = np.zeros((self.dim))
                padded[:wav.shape[0]] = wav
                X[i,] = padded[:, np.newaxis]
                y[i] = self.labels[ID]
                
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
            
        elif self.input_type == 'spectrogram':
            for i, ID in enumerate(list_IDs_temp):
                if self.labels[ID] == 11:
                    path = '../input/'
                else:
                    path = train_audio_path + '/'
                sample_rate, samples = wavfile.read(path + ID)

                if self.augment:
                    if random.randint(1,101) < 51:
                        sample_rate, samples = bandpass(sample_rate, samples)
                        
                spect = spectrogram(sample_rate, samples)
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.dim))
                padded[:spect.shape[0], :spect.shape[1]] = spect
                X[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
                
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
            
        elif self.input_type == 'all':
            for i, ID in enumerate(list_IDs_temp):
                wav = wavread(ID)
                padded = np.zeros((self.x1dim))
                padded[:wav.shape[0]] = wav
                X1[i,] = padded[:, np.newaxis]
                y[i] = self.labels[ID]
        
            for i, ID in enumerate(list_IDs_temp):
                spect = spectrogram(ID)
                padded = np.zeros((self.x2dim))
                padded[:spect.shape[0], :spect.shape[1]] = spect
                X2[i,] = padded[:, :, np.newaxis]        
                
            return X1, X2, keras.utils.to_categorical(y, num_classes=self.n_classes)
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# Parameters
train_params = {'dim': (71,129),
                'batch_size': 64,
                'n_classes':12,
                'n_channels': 1,
                'shuffle': True,
                'input_type': 'spectrogram',
                'augment': True
               }
val_params = {'dim': (71,129),
              'batch_size': 64,
              'n_classes':12,
              'n_channels': 1,
              'shuffle': False,
              'input_type': 'spectrogram',
              'augment': False
             }

# Generators
spect_training_generator = DataGenerator(partition['train'], labels, **train_params)
spect_validation_generator = DataGenerator(partition['validation'], labels, **val_params)

# Design model
if False:
    
    spect_input = keras.Input(shape=(71, 129, 1))
    x = Conv2D(32, 3, activation='relu')(spect_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 5, strides=2, padding='same', activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, 4, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)

    spect_prediction = Dense(12, activation='softmax')(x)

    spect_model = Model(spect_input, spect_prediction)

    spect_model.compile(optimizer = 'adam',
                        loss='categorical_crossentropy',
                        metrics=["accuracy"])

    spect_model.summary()

    spect_annealer = LearningRateScheduler(lambda x: 1e-3 * 0.98 ** x)

else:

    spect_annealer = LearningRateScheduler(lambda x: 1e-4 * 0.98 ** x)
    spect_model = load_model('../input/spect-model/spect_model.h5')
    
# Train model on dataset
spect_history = spect_model.fit_generator(generator=spect_training_generator,
                                          validation_data=spect_validation_generator,
                                          steps_per_epoch=300,
                                          epochs=10,
                                          verbose=2,
                                          callbacks=[spect_annealer]
                                         )
acc = spect_history.history['acc']
val_acc = spect_history.history['val_acc']
loss = spect_history.history['loss']
val_loss = spect_history.history['val_loss']

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
spect_model.save('spect_model.h5')
