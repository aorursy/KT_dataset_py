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
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM

from matplotlib import pyplot as plt

import math

from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv('/kaggle/input/rk-puram-ambient-air/trainset.csv', index_col=0, header=0)

data = data.drop(['Weekday', 'Month', 'Year', 'Hour'], axis=1)

data
# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]

	df = pd.DataFrame(data)

	cols, names = list(), list()

	# input sequence (t-n, ... t-1)

	for i in range(n_in, 0, -1):

		cols.append(df.shift(i))

		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)

	for i in range(0, n_out):

		cols.append(df.shift(-i))

		if i == 0:

			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

		else:

			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together

	agg = pd.concat(cols, axis=1)

	agg.columns = names

	# drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg
# load dataset

values = data.values

# ensure all data is float

values = values.astype('float32')

# normalize features

print(values.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)

# frame as supervised learning

reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict

reframed.drop(reframed.columns[[15,16,17,18,19,20]], axis=1, inplace=True)

print(reframed.head())
# split into train and test sets

values = reframed.values

n_train_hours = 365 * 24

train = values[:n_train_hours, :]

test = values[n_train_hours:, :]

# split into input and outputs

train_X, train_y = train[:, :15], train[:, 15:]

test_X, test_y = test[:, :15], test[:, 15:]

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network

model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(9))

model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
# make a prediction

yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X[:, :6]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,:9]

# invert scaling for actual

test_y = test_y.reshape((len(test_y), 9))

inv_y = np.concatenate((test_y, test_X[:, :6]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,:9]

# calculate RMSE

for x in range(9):

    rmse = sqrt(mean_squared_error(inv_y[:, x], inv_yhat[:, x]))

    print('Test RMSE of %s: %.3f' % (data.columns[6+x], rmse))
for x in range(9):

    rmse = mean_absolute_error(inv_y[:, x], inv_yhat[:, x])

    print('Test MAE of %s: %.3f' % (data.columns[6+x], rmse))