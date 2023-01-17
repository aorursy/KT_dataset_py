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
import sys

import numpy

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import LSTM

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils
# load ascii text and covert to lowercase

filename = "/kaggle/input/peterpan.txt"

raw_text = open(filename, 'r', encoding='utf-8').read()

raw_text = raw_text.lower()


# create mapping of unique chars to integers

chars = sorted(list(set(raw_text)))

char_to_int = dict((c, i) for i, c in enumerate(chars))
n_chars = len(raw_text)

n_vocab = len(chars)

print("Total Characters: ", n_chars)

print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers

seq_length = 100

dataX = []

dataY = []

for i in range(0, n_chars - seq_length, 1):

    seq_in = raw_text[i:i + seq_length]

    seq_out = raw_text[i + seq_length]

    dataX.append([char_to_int[char] for char in seq_in])

    dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)

print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]

X = numpy.reshape(dataX, (n_patterns, seq_length, 1))

# normalize

X = X / float(n_vocab)

# one hot encode the output variable

y = np_utils.to_categorical(dataY)
# define the LSTM model

model = Sequential()

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))

model.add(Dropout(0.2))

model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]
model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)
# load the network weights

filename = "weights-improvement-20-2.0357.hdf5"

model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
int_to_char = dict((i, c) for i, c in enumerate(chars))
# pick a random seed

start = numpy.random.randint(0, len(dataX)-1)

pattern = dataX[start]

print("Seed:")

print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# generate characters

for i in range(1000):

    x = numpy.reshape(pattern, (1, len(pattern), 1))

    x = x / float(n_vocab)

    prediction = model.predict(x, verbose=0)

    index = numpy.argmax(prediction)

    result = int_to_char[index]

    seq_in = [int_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)

    pattern = pattern[1:len(pattern)]

print("\nDone.")