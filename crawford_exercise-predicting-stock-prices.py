import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.time_series_with_siraj.ex1 import *

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

print("Setup Complete")

X_train, y_train, X_test, y_test = load_data('../input/sp500.csv', 50, True)

# Your exploratory data analysis code below
siraj_model = Sequential()

siraj_model.add(LSTM(
    input_shape=(None, 1),
    units=50,
    return_sequences=True))
siraj_model.add(Dropout(0.2))

siraj_model.add(LSTM(100, return_sequences=False))
siraj_model.add(Dropout(0.2))

siraj_model.add(Dense(1))
siraj_model.add(Activation('linear'))

siraj_model.compile(loss='mse', optimizer='rmsprop')


siraj_model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05)

step_1.check()
my_model = Sequential()

my_model.add(LSTM(
    input_shape=(None, 1),
    units=50,
    return_sequences=True))


my_model.add(LSTM(100, return_sequences=False))
my_model.add(Dropout(0.2))

my_model.add(Dense(1))
my_model.add(Activation('linear'))

my_model.compile(loss='mse', optimizer='rmsprop')


my_model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=1,
    validation_split=0.05)

# Fill in the parameters to fit your model
my_model.fit(
    X_train,
    y_train,
    batch_size=____,   # Fill this in
    epochs=____,       # Fill this in
    validation_split=0.05)

step_2.check()