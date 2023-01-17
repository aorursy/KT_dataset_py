from __future__ import print_function

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

np.random.seed(2)  # for reproducibility

from keras.preprocessing import sequence

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Embedding

from keras.layers import LSTM, SimpleRNN, GRU

from keras.datasets import imdb

from keras.utils.np_utils import to_categorical

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)

from sklearn import metrics

from sklearn.preprocessing import Normalizer

import h5py

from keras import callbacks

import tensorflow as ft

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

train= pd.read_csv('../input/kddcup99/Training.csv')

test = pd.read_csv('../input/kddcup99/Testing.csv')
X = train.iloc[:,1:21]

Y = train.iloc[:,0]

C = test.iloc[:,0]

T = test.iloc[:,1:21]

trainX = np.array(X)

testT = np.array(T)

trainX.astype(float)

testT.astype(float)

scaler = Normalizer().fit(trainX)

trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)

testT = scaler.transform(testT)

y_train = np.array(Y)

y_test = np.array(C)

X_train = np.array(trainX)

X_test = np.array(testT)

import tensorflow as tf

from tensorflow import keras



model = tf.keras.Sequential()

# Add an Embedding layer expecting input vocab of size 1000, and

# output embedding dimension of size 64.

model.add(keras.layers.Embedding(input_dim=20, output_dim=50))



# Add a LSTM layer with 128 internal units.

model.add(keras.layers.LSTM(50))



# Add a Dense layer with 10 units and softmax activation.

model.add(keras.layers.Dense(1, activation='softmax'))



model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, nb_epoch=50)