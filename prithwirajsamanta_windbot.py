# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.models import load_model

import types

import os

from contextlib import suppress
# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

	n_vars = 1 if type(data) is list else data.shape[1]

	df = DataFrame(data)

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

	agg = concat(cols, axis=1)

	agg.columns = names

    

	# drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg
df = pd.read_csv('../input/wind-turbine-scada-dataset/T1.csv', header=0, index_col=0)

columnsTitles=['LV ActivePower (kW)','Wind Speed (m/s)',"Wind Direction (°)","Theoretical_Power_Curve (KWh)"]

df=df.reindex(columns=columnsTitles)

df.head()
# load dataset

df['LV ActivePower (kW)']=df['LV ActivePower (kW)'].div(5000)

df['Wind Speed (m/s)']=df['Wind Speed (m/s)'].div(30)

df['Theoretical_Power_Curve (KWh)']=df['Theoretical_Power_Curve (KWh)'].div(5000)

df['Wind Direction (°)']=df['Wind Direction (°)'].div(360)

dataset = df

values = dataset.values

#print(values)



# specify the number of lag and ahead hours

n_hours = 24

n_ahead = 1

n_features = 4



# integer encode direction

#encoder = LabelEncoder()

#values[:,n_features-1] = encoder.fit_transform(values[:,n_features-1])



# ensure all data is float

values = values.astype('float32')

#print(df)



# normalize features

#scaler = MinMaxScaler(feature_range=(0, 1))

#scaled = scaler.fit_transform(values)

#print(scaled[0:30])



# frame as supervised learning

reframed = series_to_supervised(values, n_hours,n_ahead, 1)

#print(reframed[0:24])

print(reframed.shape)
# split into train and test sets

values = reframed.values

n_train_hours = (int)(len(dataset)*0.999)

train = values[:n_train_hours, :]

test = values[n_train_hours:, :]



#print(train)



# split into input and outputs

n_obs = n_hours * n_features

train_X, train_y = train[:, :n_obs], train[:, -n_features]

test_X, test_y = test[:, :n_obs], test[:, -n_features]

print(train_X.shape, len(train_X), train_y.shape)



#print(train_X)

#print(train_y)



# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))

test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()

model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
# fit network

history = model.fit(train_X, train_y, epochs=150, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False)

model.save("WindBot.h5")
filename = 'WindBot'

# Delete a duplicate file if exists.

with suppress(OSError):

    os.remove(filename)

model.save(filename+".h5",overwrite=True)



#compress keras model

tar_filename = filename + '.tgz'

cmdstring = 'tar -zcvf ' + tar_filename + ' ' + filename+".h5"

print(cmdstring)

os.system(cmdstring)
model = load_model('WindBot.h5')
# plot history

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()



#Copying test data

test_C=test_X

test_X=test_C

#yhat = model.predict(test_X)

#test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

#print(test_X)



# invert scaling for forecast

#inv_yhat = concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)

#inv_yhat = scaler.inverse_transform(inv_yhat)

#inv_yhat = inv_yhat[:,0]



inv_yhat=model.predict(test_X)

for i in range(len(inv_yhat)):

    inv_yhat[i]=inv_yhat[i]*5000

#print(inv_yhat)



# invert scaling for actual

#test_y = test_y.reshape((len(test_y), 1))

#inv_y = concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)

#inv_y = scaler.inverse_transform(inv_y)

#inv_y = inv_y[:,0]



inv_y=test_y

for i in range(len(inv_y)):

    inv_y[i]=inv_y[i]*5000

#print(inv_y)



# calculate RMSE

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
# plot history

pyplot.plot(inv_yhat, label='predicted')

pyplot.plot(inv_y, label='true')

pyplot.legend()

pyplot.show()