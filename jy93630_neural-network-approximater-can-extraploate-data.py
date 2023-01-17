import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import optimizers
import math
import seaborn as sns
x = np.linspace(-10, 10, 100)
y = np.cos(x)
plt.plot(x,y)
plt.show()
model = Sequential()
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(1, activation='linear'))
ada = optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=ada, metrics=['mean_squared_error'])
model.fit(x, y, epochs=100, verbose=0)
x = np.linspace(-10, 10, 100)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y) 
plt.plot(x, prediction)
plt.show()
x = np.linspace(-20, 20, 200)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y)
plt.plot(x, prediction)
plt.show()
x = np.linspace(-10, 10, 100)
y = np.cos(x)

from keras import regularizers
model = Sequential()
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))
ada = optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=ada, metrics=['mean_squared_error'])
model.fit(x, y, epochs=100, verbose=0)
x = np.linspace(-20, 20, 200)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y)
plt.plot(x, prediction)
plt.show()