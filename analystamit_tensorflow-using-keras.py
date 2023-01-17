from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')

data_train.shape, data_test.shape
X_train = np.array(data_train.iloc[:, 1:]).astype('float32')

y_train = np.array(data_train.iloc[:, 0])

X_test = np.array(data_test.iloc[:, 1:]).astype('float32')

y_test = np.array(data_test.iloc[:, 0])

X_train = X_train / 255

X_test = X_test / 255
np.unique(y_train)
n_classes = 10

y_train = keras.utils.to_categorical(y_train, n_classes)

y_test = keras.utils.to_categorical(y_test, n_classes)
model = Sequential()

model.add(Dense(64, activation='sigmoid', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))
model.summary()

print((64*784)+64)

print((10*64) + 10)
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, verbose = 0, epochs=40, validation_data=(X_test, y_test))

model = Sequential()

model.add(Dense(64, activation='tanh', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784,)))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_test, y_test))