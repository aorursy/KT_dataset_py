import numpy as np

import pandas as pd

import keras

import tensorflow as tf

from keras.layers import Dense, MaxPool2D, Flatten, Dropout, BatchNormalization, Input

from keras.models import Sequential

from keras.utils import data_utils

import sklearn

from keras.metrics import mse

from sklearn.datasets import load_boston
data = load_boston()
X = data.data

Y = data.target
from sklearn.model_selection import train_test_split as tts

xtrain,xtest, ytrain,ytest = tts(X,Y)

xtrain.shape, xtrain[0]
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

# dir(scaler)

xtrain = scaler.fit_transform(xtrain)

xtest = scaler.transform(xtest)
model = Sequential()

model.add(Dense(1000, activation = 'relu', input_dim = 13))

model.add(Dropout(0.1))

model.add(Dense(1000, activation = 'relu'))

model.add(Dropout(0.1))

model.add(Dense(1)) # no activation



model.compile('adam', loss = 'mse', metrics = [mse])

model.summary()
hist= model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs = 300)
model.evaluate(xtest, ytest)
import matplotlib.pyplot as plt







#adam



plt.figure(1)

# plt.plot(hist.history['acc'], color = 'r')

# plt.plot(hist.history['val_acc'], color = 'b')



plt.figure(2)

plt.plot(hist.history['loss'], color = 'r')

plt.plot(hist.history['val_loss'], color = 'b')

plt.xlabel('loss')

plt.ylabel('epochs')

plt.title('loss curve')

from sklearn.metrics import r2_score

r2_score(ytest, model.predict(xtest))