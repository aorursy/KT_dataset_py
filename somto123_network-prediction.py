class StrToBytes:

    def __init__(self, fileobj):

        self.fileobj = fileobj

    def read(self, size):

        return self.fileobj.read(size).encode()

    def readline(self, size=-1):

        return self.fileobj.readline(size).encode()



# -*- coding: utf-8 -*-

def normalize_data(x_train, x_test):

	"""

	$x_train and $x_test are numpy arrays which should be normalized.

	Normalizes the training data to have a train_mean of 0 and a standard deviation of 1. 

	The test data is normalized with the parameters of the training data

	Returns the normalized data in the same format as given.

	"""

	train_mean = numpy.mean(x_train)

	train_std_dev = numpy.std(x_train)

	x_train = (x_train - train_mean) / train_std_dev # element-wise operations

	x_test = (x_test - train_mean) / train_std_dev # element-wise operations

	return x_train, x_test
def shuffle_in_unison(x_data, y_data, seed=195735):

	'''

	This method shuffles the data of the Data Set axis of the test and training data.

	It is inspired by http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison

	'''

	numpy.random.seed(seed)

	shuffled_x = numpy.empty(x_data.shape, dtype=x_data.dtype)

	shuffled_y = numpy.empty(y_data.shape, dtype=y_data.dtype)

	permutation = numpy.random.permutation(x_data.shape[2])

	for old_index, new_index in enumerate(permutation):

		shuffled_x[::,::,new_index,::,::] = x_data[::,::,old_index,::,::]

		shuffled_y[::,::,new_index] = y_data[::,::,old_index]

	return shuffled_x, shuffled_y
def split_data(x_data, y_data, validation_fraction=0.2):

	"""

	Splits the data into training and validation data

	according to the fraction that was specified. The samples are shuffled and then selected.

    The data is equally splitted along classes and signal to noise ratios.

	The new data array, validation array and the according label arrays are returned.

	"""

	# Shuffle data

	x_data, y_data = shuffle_in_unison(x_data, y_data)

	# Split data

	nb_sets = x_data.shape[2]

	nb_cutted = int(numpy.floor(nb_sets * validation_fraction))

	x_test = x_data[::,::,-1:(-nb_cutted-1):-1,::,::]

	y_test = y_data[::,::,-1:(-nb_cutted-1):-1]

	x_data = numpy.delete(x_data, numpy.s_[-1:(-nb_cutted-1):-1], axis=2)

	y_data = numpy.delete(y_data, numpy.s_[-1:(-nb_cutted-1):-1], axis=2)

	return x_data, y_data, x_test, y_test

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pickle

from keras import backend as K

if K.backend() == 'theano':

    K.set_image_dim_ordering('tf')

import os

which_kind='iq'

print(os.listdir("../input"))

data = []

# Any results you write to the current directory are saved as output.
x_data = None

y_data = None

data_file = None

# decide which data to load

if which_kind == 'iq':

    data_file = 'data_iq.p'

elif which_kind == 'fft':

    data_file = 'data_fft.p'

else:

    raise ValueError('Parameter which_kind must be "iq" for IQ-data or "fft" for FFT-data.')

# load input data (x)

data_path = os.path.join('../input', data_file)

print(data_path)

with open(data_path, mode='r') as storage: 

    x_data = pickle.load(StrToBytes(storage), encoding='latin1')

    # load output data/labels (y)

label_file = 'labels.p'

label_path = os.path.join('../input', label_file)

with open(label_path, mode='rb') as storage:

    y_data = pickle.load(storage)

# return x_data, y_data
print(y_data.shape)

print(x_data.shape)
x_data, y_data, x_test, y_test = split_data(x_data, y_data, validation_fraction=0.2)

y_data.shape

print(len(y_data))
x_data, x_test = normalize_data(x_data, x_test )
for i in range(len(x_data)):

    for x in range(len(x_data[i])):

        for j in range(len(x_data[i][x])):

            data.append(x_data[i][x][j])

y_data = y_data.flatten()

data = np.array(data)
data.shape
from keras.models import Sequential

from keras.layers import Dense, Activation,Conv2D,Dropout,Flatten

import keras
model = Sequential()

model.add(Dense(100, input_shape=( 128, 2)))

# model.add(Conv2D(8, kernel_size=(3,1) ,padding='same',activation='relu', input_shape=( 128, 2,1)))

model.add(Flatten())

model.add(Activation('relu'))

model.add(Dense(100, activation='sigmoid'))

model.add(Dense(59, activation='relu'))

model.add(Dense(10, activation='sigmoid'))

model.add(Dense(15, activation='linear'))

# model.add(Conv2D(8, kernel_size=(3,1) ,padding='same',activation='relu', input_shape=( 128, 2,1)))

# model.add(Conv2D(16, kernel_size=(3,2) ,activation='relu'))

# model.add(Activation('relu'))

# model.add(Dropout(0.6))

# # model.add(Flatten())

# model.add(Dense(128,activation='relu'))

# model.add(Dropout(0.6))

# model.add(Dense(15,activation='softmax'))
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)



model.compile(loss='categorical_hinge',

              optimizer=opt,

              metrics=['accuracy'])

batch_size = 32

num_classes = 10

epochs = 100

data_augmentation = True

num_predictions = 15

model.summary()
from keras.utils import to_categorical

labels = to_categorical(y_data)

print(labels.shape)

model.fit(data, labels,

              batch_size=batch_size,

              epochs=epochs,

              shuffle=True)