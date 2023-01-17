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
# Import necessary library

import time

import pandas as pd

from datetime import datetime  

from datetime import timedelta 

import io

import requests

from matplotlib import pyplot

from math import isnan

from math import sqrt

from numpy import concatenate

import numpy as np

from pandas import concat

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.utils import plot_model

from keras.models import load_model

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from IPython.display import clear_output

from random import randint



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

	agg = concat(cols, axis=1)

	agg.columns = names

	# drop rows with NaN values

	if dropnan:

		agg.dropna(inplace=True)

	return agg
#Import gas heater data

data2014=pd.read_csv('/kaggle/input/gasheater/2014.csv',index_col ='time')

data2015=pd.read_csv('/kaggle/input/gasheater/2015.csv',index_col ='time')

data2016=pd.read_csv('/kaggle/input/gasheater/2016_1.csv',index_col ='time')

df=data2014.append(data2015).append(data2016)



#Normalize Dataframe

values = data2014.values

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



# specify the number of lag hours

n_hours = 8

n_features = 13

# frame as supervised learning

reframed = series_to_supervised(scaled, n_hours, 6)

print(reframed.shape)

# split into train and test sets

values = reframed.values[:600,:]

n_train_hours = int(values.shape[0]*0.5) #Train 80% of data

train = values[:n_train_hours, :]

test = values[(n_train_hours):, :]

# split into input and outputs

n_obs = n_hours * n_features

train_X, train_y = train[:, :n_obs], train[:,-n_features*1]

test_X, test_y = test[:, :n_obs], test[:, -n_features*1]

print(train_X.shape, len(train_X), train_y.shape)



# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))

test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network

model = Sequential()

model.add(LSTM(8, return_sequences=True, stateful=True, batch_size = 4,

                    input_shape=(train_X.shape[1],train_X.shape[2]),

                    dropout = 0.5))

model.add(LSTM(8, return_sequences=False, recurrent_dropout=0.5))

#model.add(Dropout(0.2))

#model.add(LSTM(1, return_sequences=False))

#model.add(LSTM(8, return_sequences=False))

#model.add(Dense(2, activation='relu'))

#model.add(Dense(8, activation='relu'))

#model.add(Dense(8, activation='sigmoid'))

#model.add(Dense(3, activation='exponential'))

#model.add(Dense(4, activation='exponential'))

#model.add(Dense(4, activation='sigmoid'))

#model.add(Dense(4))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')



loss=[]

val_loss=[]
#fit network



epoch_number = 10000

max_val_loss = 0.05



for m in range(epoch_number):

            

    history = model.fit(train_X, train_y, epochs=1, batch_size=4, validation_data=(test_X, test_y),

                            verbose=1, shuffle=False)

    model.reset_states()

    loss = np.append(loss,history.history['loss'])

    val_loss = np.append(val_loss,history.history['val_loss'])

    clear_output()

    

    print("epoch no = ", str(m))

    # plot history

    plt.figure(figsize=(16,8))

    pyplot.plot(loss, label='train')

    pyplot.plot(val_loss, label='test')

    pyplot.legend()

    pyplot.show()

    

    if list(history.history['val_loss'])[0]<max_val_loss:

            break
#

# make a prediction

yhat = model.predict(test_X, batch_size=4)



# invert scaling for forecast

test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))



inv_yhat=np.zeros((300,13))

inv_yhat[:295,-12:]=test_X[5:,-12:]

inv_yhat[:,0]=list(yhat)[:]

print(inv_yhat[0,:])



#inv_yhat = concatenate((yhat, test_X[:,-12:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

print(inv_yhat[0,:])



inv_yhat = inv_yhat[:,0]



# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))



inv_y=np.zeros((300,13))

inv_y[:295,-12:]=test_X[5:,-12:]

inv_y[:,0]=list(test_y)[:]

print(inv_y[0,:])



#inv_y = concatenate((test_y, test_X[:, -12:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

print(inv_y[0,:])



inv_y = inv_y[:,0]



# calculate RMSE

rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)



plt.figure(figsize=(16,8))

pyplot.plot(inv_y,label='actual')

pyplot.plot(inv_yhat,label='predict')

#pyplot.plot(whole_data.values[:,0],label='train')

#pyplot.yscale("log")

pyplot.legend()

pyplot.show()

























list(history.history['loss'])[0]

model.save("model_gasheater.h5")

print("Saved model to disk")
# load model

model = load_model("model_gasheater.h5")