# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, GRU, Bidirectional, LSTM

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error

import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = [1,2,3,4,5,6]



datanp = np.array(data)
datanp.shape
sample = datanp.reshape(3,2)
x_train = np.array([1,2,2,3,3,4,4,5])

y_train = datanp[2:6]
x_train = x_train.reshape(4,1,2)

x_train
y_train
def make_gru_network():

    regressorGRU = Sequential()

    # First GRU layer with Dropout regularisation

    

    regressorGRU.add(GRU(units = 30, return_sequences = False, input_shape=(1,2), activation='tanh'))

    regressorGRU.add(Dropout(0.3))

    Sequential

    # The output layer

    regressorGRU.add(Dense(units=1))

    return regressorGRU
model = make_gru_network()
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 10, batch_size = 2)
x = np.array([[[4,5]]])

x.shape
model.predict(x)