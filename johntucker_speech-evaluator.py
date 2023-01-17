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
import pickle
import keras

from pathlib import Path
from subprocess import check_output

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
with open('../input/speech-recognition-data-processing/SavedTestDict.pickle', 'rb') as handle:
    test_dict = pickle.load(handle)
with open('../input/speech-recognition-data-processing/SavedTestLabels.pickle', 'rb') as handle:
    test_labels = pickle.load(handle)
print("loaded data.")
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio'
contest_dict = {'yes': 0,
                'no': 1,
                'up': 2,
                'down': 3,
                'left': 4,
                'right': 5,
                'on': 6,
                'off': 7,
                'stop': 8,
                'go': 9,
                'unknown': 10,
                'silence': 11
               }
answer_dict = {0: 'yes',
               1: 'no', 
               2: 'up', 
               3: 'down', 
               4: 'left',
               5: 'right',
               6: 'on',
               7: 'off',
               8: 'stop',
               9: 'go',
               10: 'unknown',
               11: 'silence'
              }
print(test_dict['test'][0], answer_dict[test_labels[test_dict['test'][0]]])
def spectrogram(file, label):
    if label == 11:
        path = '../input/'
    else:
        path = train_audio_path + '/'
        
    eps=1e-10
    sample_rate, samples = wavfile.read(path + file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    
    # silence can end up being empty files, in this case we can just return one second of zeros
    if len(spectrogram.shape) < 2:
        return np.zeros((71,129))
    else:
        return np.log(np.abs(spectrogram).T+eps)

def stft(file, label):
    if label == 11:
        path = '../input/'
    else:
        path = train_audio_path + '/'
        
    eps=1e-10
    sample_rate, samples = wavfile.read(path + file)
    frequencies, times, Zxx = signal.stft(samples, sample_rate, nperseg = sample_rate/50, noverlap = sample_rate/75)
    
    # silence can end up being empty files, in this case we can just return one second of zeros
    if len(Zxx.shape) < 2:
        return np.zeros((151,161))
    else:
        return np.log(np.abs(Zxx).T+eps)
class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_IDs, labels, batch_size=32, dim=(151,161), x1dim=(71,129), x2dim=(151,161), n_channels=1,
                 n_classes=11, shuffle=True, input_type='wav', testing=False):
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
                spect = spectrogram(ID, self.labels[ID])
                padded = np.zeros((self.dim))
                padded[:spect.shape[0], :spect.shape[1]] = spect
                X[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
                
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

        elif self.input_type == 'stft':
            for i, ID in enumerate(list_IDs_temp):
                trans = stft(ID, self.labels[ID])
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X[i,] = padded[:, :, np.newaxis]
                y[i] = self.labels[ID]
                
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        
        elif self.input_type == 'all':
        
            for i, ID in enumerate(list_IDs_temp):
                spect = spectrogram(ID, self.labels[ID])
                padded = np.zeros((self.x1dim))
                padded[:spect.shape[0], :spect.shape[1]] = spect
                X1[i,] = padded[:, :, np.newaxis]        
                y[i] = self.labels[ID]
            for i, ID in enumerate(list_IDs_temp):
                trans = stft(ID, self.labels[ID])
                #last = ID, self.dim, spect.shape

                padded = np.zeros((self.dim))
                padded[:trans.shape[0], :trans.shape[1]] = trans
                X2[i,] = padded[:, :, np.newaxis]
                
            return X1, X2, keras.utils.to_categorical(y, num_classes=self.n_classes)
from keras.models import load_model

ensemble_model = load_model('../input/ensemble-model/ensemble_model.h5')
# Parameters
test_params = {'x1dim': (71,129),
               'x2dim': (151,161),
               'batch_size': 5,
               'n_classes':12,
               'n_channels': 1,
               'shuffle': False,
               'input_type': 'all'}

# Generators
test_generator = DataGenerator(test_dict['test'], test_labels, **test_params)

ensemble_model.evaluate_generator(test_generator, steps=1367)
