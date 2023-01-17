# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Importing libraries
from pathlib import Path
from sklearn.utils import shuffle
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Step 1 Preprocessing
# Converting wav to dataframes using log_spectogram

def get_data(path):
    datadir = Path(path)
    files = [(str(f), f.parts[-2]) for f in datadir.glob('**/*.wav') if f]
    dataset = pd.DataFrame(files, columns=['path', 'label'])
    return dataset

# Preparing the dataset as described in the dataset
def prepare_data(dataset):
    train_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
    words = dataset.label.unique().tolist()
    silence = ['_background_noise_']
    unknown = [w for w in words if w not in silence + train_words]

    # there are only 6 silence files. Mark them as unknown 
    dataset.loc[dataset.label.isin(silence), 'label'] = 'unknown'
    dataset.loc[dataset.label.isin(unknown), 'label'] = 'unknown'
    return dataset

train = prepare_data(get_data("../input/train/audio/"))
# Spectogram function 
def log_spectogram(audio, sample_rate, window_size=10,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    _, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return np.log(spec.T.astype(np.float32) + eps)
# Getting all the paths to the files
paths = []
files = train['path']
for i in range(len(files)):
    path = str(files[i])
    paths.append(path)
    
def audio_to_data(path):
    # we take a single path and convert it into data
    sample_rate, audio = wavfile.read(path)
    spectrogram = log_spectogram(audio, sample_rate, 10, 0)
    return spectrogram.T

def data_generator(paths,labels):
    data = np.zeros(shape = (len(paths), 81, 100))
    indexes = []
    for i in tqdm(range(len(paths))):
        audio = audio_to_data(paths[i])
        if audio.shape != (81,100):
            indexes.append(i)
        else:
            data[i] = audio
    final_labels = [l for i,l in enumerate(labels) if i not in indexes]
    print('Number of instances with inconsistent shape:', len(indexes))
    return data[:len(data)-len(indexes)], final_labels, indexes
# Converting labels to Bitmap using LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()
y = labelbinarizer.fit_transform(train.label)

# Obtaining data
data,l,indexes = data_generator(paths,y)

labels = np.zeros(shape = [data.shape[0], 81, len(l[0])])

input_length = np.zeros([32, 1])
label_length = np.zeros([32, 1])

for i,array in enumerate(l):
    for j, element in enumerate(array):
        labels[i][j] = element
        
print(data.shape)
print(labels.shape)

data,labels = shuffle(data,labels)

print(data[0].shape)
print(labels[0].shape)
# Splitting the converted dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
# Part 2 - Building the model

#Importing the Keras libraries and packages
from keras.layers import Dropout, Dense
from keras.layers import Bidirectional, SimpleRNN, Lambda, Input, TimeDistributed
from keras.models import Sequential,Model
import tensorflow as tf
from keras import backend as K
from keras.activations import relu

# Clipped ReLu function as described in paper
def clipped_relu(x):
    return relu(x, max_value=20)

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"clipped_relu": clipped_relu})
K.set_learning_phase(1)  
model = Sequential()

# Layer 1 with clipped ReLu activation function
model.add(Dense(512, activation = clipped_relu, input_shape=(81,100)))
model.add(Dropout(rate = 0.1))

# Layer 2 with clipped ReLu activation function
model.add(Dense(256, activation = clipped_relu))
model.add(Dropout(rate = 0.1))

# Layer 3 with clipped ReLu activation function
model.add(Dense(256, activation = clipped_relu))
model.add(Dropout(rate = 0.1))

# Layer 4 Bidirectional Recurrent layer with clipped ReLu activtion function
model.add(Bidirectional(SimpleRNN(512, activation = clipped_relu, return_sequences = True)))
model.add(Dropout(rate = 0.1))

# Layer 5 with softmax activaiton function
model.add(Dense(units = 11, activation = "softmax"))

# Compiling the model
model.compile(optimizer = 'adam', loss = "binary_crossentropy")
model.summary()

# Fitting the model
model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test,y_test))