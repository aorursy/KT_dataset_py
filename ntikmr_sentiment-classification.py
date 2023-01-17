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
#we can load the data from dataset api in keras

import tensorflow as tf

from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)

print("Training set")

print(x_train.shape)

print(y_train.shape)

print("Test set")

print(x_test.shape)

print(y_test.shape)
print(x_train[0])

print(y_train[0])
#let's build the model

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.layers import Dense, Conv1D, Activation, Dropout, Flatten, MaxPooling1D, LSTM

from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping

#pad the reviews to a maximum length of 500 words

max_length = 500

x_train = sequence.pad_sequences(x_train, max_length)

x_test = sequence.pad_sequences(x_test, max_length)



#create the model

model = Sequential()

model.add(Embedding(10000, 64, input_length = max_length))

model.add(Dropout(0.2))

model.add(Conv1D(64, 3, padding = 'same', kernel_initializer = 'he_normal'))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.5))



model.add(LSTM(100))





model.add(Dense(512, kernel_initializer = 'he_normal'))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(128, kernel_initializer = 'he_normal'))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))

model.summary()
callbacks = [ModelCheckpoint('imdb_sentiment.h5', monitor = 'val_loss', save_best_only = True, mode = 'auto'),

             EarlyStopping(monitor = 'val_loss', patience= 5)]

model.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, batch_size = 64, epochs = 20, validation_data = (x_test, y_test), callbacks = callbacks)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()