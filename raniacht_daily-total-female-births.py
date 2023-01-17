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
import pandas as pd

dailytotalfemalebirths = pd.read_csv("../input/dailytotalfemalebirths.csv")


# univariate mlp example

from numpy import array

from keras.models import Sequential

from keras.layers import Dense

# define dataset

X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([40, 50, 60, 70])

# define model

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=3))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=2000, verbose=0)

# demonstrate prediction

x_input = array([50, 60, 70])

x_input = x_input.reshape((1, 3))

yhat = model.predict(x_input, verbose=0)

print(yhat)





# univariate cnn example

from numpy import array

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

# define dataset

X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([40, 50, 60, 70])

# reshape from [samples, timesteps] into [samples, timesteps, features]

X = X.reshape((X.shape[0], X.shape[1], 1))

# define model

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction

x_input = array([50, 60, 70])

x_input = x_input.reshape((1, 3, 1))

yhat = model.predict(x_input, verbose=0)

print(yhat)
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

# define dataset

X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([40, 50, 60, 70])

# reshape from [samples, timesteps] into [samples, timesteps, features]

X = X.reshape((X.shape[0], X.shape[1], 1))

# define model

model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(3, 1)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction

x_input = array([50, 60, 70])

x_input = x_input.reshape((1, 3, 1))

yhat = model.predict(x_input, verbose=0)

print(yhat)
# univariate cnn-lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import TimeDistributed

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

# define dataset

X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])

y = array([50, 60, 70, 80])

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]

X = X.reshape((X.shape[0], 2, 2, 1))

# define model

model = Sequential()

model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, 2, 1)))

model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

model.add(TimeDistributed(Flatten()))

model.add(LSTM(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=500, verbose=0)

# demonstrate prediction

x_input = array([50, 60, 70, 80])

x_input = x_input.reshape((1, 2, 2, 1))

yhat = model.predict(x_input, verbose=0)

print(yhat)
from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import RepeatVector

from keras.layers import TimeDistributed

# define dataset

X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([[40,50],[50,60],[60,70],[70,80]])

# reshape from [samples, timesteps] into [samples, timesteps, features]

X = X.reshape((X.shape[0], X.shape[1], 1))

y = y.reshape((y.shape[0], y.shape[1], 1))

# define model

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(3, 1)))

model.add(RepeatVector(2))

model.add(LSTM(100, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(1)))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=100, verbose=0)

# demonstrate prediction

x_input = array([50, 60, 70])

x_input = x_input.reshape((1, 3, 1))

yhat = model.predict(x_input, verbose=0)

print(yhat)