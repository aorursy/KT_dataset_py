import numpy as np

import keras

import tensorflow as tf

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras_tqdm import TQDMNotebookCallback

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

import random

from random import randint
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)



def plot_multi_graph(xAxis,yAxes,title='',xAxisLabel='number',yAxisLabel='Y'):

    linestyles = ['-', '--', '-.', ':']

    plt.figure()

    plt.title(title)

    plt.xlabel(xAxisLabel)

    plt.ylabel(yAxisLabel)

    for key, value in yAxes.items():

        plt.plot(xAxis, np.array(value), label=key, linestyle=linestyles[randint(0,3)])

    plt.legend()
# define input sequence

raw_seq = [i for i in range(100)]



# Try the following if randomizing the sequence:

# random.seed('sam') # set the seed

# raw_seq = random.sample(raw_seq, 100)



# choose a number of time steps for sliding window from data start to target start

sliding_window = 20



# split into samples

X, y = split_sequence(raw_seq, sliding_window)



print(X)

print(y)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

n_features = 1

n_seq = 20

n_steps = 1

X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))

model.add(TimeDistributed(MaxPooling1D(pool_size=1)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# fit model

history = model.fit(X, y, epochs=100, verbose=1, validation_data=(X,y))
#Plot Error

# Mean Square Error

yAxes = {}

yAxes["Training"]=history.history['mean_squared_error']

yAxes["Validation"]=history.history['val_mean_squared_error']

plot_multi_graph(history.epoch,yAxes, title='Mean Square Error',xAxisLabel='Epochs')
# demonstrate prediction

x_input = array([i for i in range(100,120)])

print(x_input)

x_input = x_input.reshape((1, n_seq, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# demonstrate prediction in data

yhat = model.predict(X, verbose=0)

print(yhat)
print(y)
xAxis = [i for i in range(len(y))]

yAxes = {}

yAxes["Data"]=raw_seq[0:len(raw_seq)-sliding_window]

yAxes["Target"]=y

yAxes["Prediction"]=yhat

plot_multi_graph(xAxis,yAxes,title='')