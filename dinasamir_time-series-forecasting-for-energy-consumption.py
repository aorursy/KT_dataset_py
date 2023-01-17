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
import numpy as np

from numpy import concatenate

import urllib.request as urllib

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error  

from keras.models import Sequential

from keras.layers import Dense



import seaborn as sns

import matplotlib.pyplot as plt

from math import sqrt



from sklearn.metrics import mean_squared_error,mean_absolute_error

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

data = pd.read_csv('/kaggle/input/hourly-energy-consumption/PJME_hourly.csv',index_col=[0], parse_dates=[0])

data.head()
import seaborn as sns

# Use seaborn style defaults and set the default figure size

sns.set(rc={'figure.figsize':(11, 4)})

data['PJME_MW'].plot(linewidth=0.5);
def create_features(df, label=None):

    """

    Creates time series features from datetime index.

    """

    df = df.copy()

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    if label:

        y = df[label]

        return X, y

    return X



X, y = create_features(data, label='PJME_MW')



df = pd.concat([X, y], axis=1)

df.head()
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

	#if dropnan:

		#agg.dropna(inplace=True)

	return agg

 
# Preprocess data

labelEncoder = LabelEncoder()

oneHotEncoder = OneHotEncoder(categorical_features=[0])

ss = StandardScaler()



values = df.values

# integer encode direction

#encoder = LabelEncoder()

#values[:,8] = encoder.fit_transform(values[:,8])

# ensure all data is float

values = values.astype('float32')

# normalize features

scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(values)



# frame as supervised learning

reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict

reframed.drop(reframed.columns[[9,10,11,12,13,14,15,16]], axis=1, inplace=True)

print(reframed.shape)

print(reframed.head())
# split into train and test sets

reframed['date_time'] = df.index.values

split_date = '01-Jan-2015'



train = reframed.loc[reframed['date_time']<=split_date].drop(['date_time'],axis=1).dropna().values

test = reframed.loc[reframed['date_time']>split_date].drop(['date_time'],axis=1).dropna().values



# split into input and output

X_train, y_train = train[:, 0:-1], train[:, -1]

X_test, y_test = test[:, 0:-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# design network

model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Dropout(0.2))

##model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(X_train, y_train, epochs=20, batch_size=70, validation_data=(X_test, y_test), verbose=2, shuffle=False)
from matplotlib import pyplot

# plot history

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()
# make a prediction

yhat = model.predict(X_test)

X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast

inv_yhat = concatenate((X_test[:,:-1],yhat), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,-1]

# invert scaling for actual

y_test = y_test.reshape((len(y_test), 1))

inv_y = concatenate((X_test[:,:-1],y_test), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,-1]

# calculate RMSE

MSE=mean_squared_error(inv_y,inv_yhat)

MAE=mean_absolute_error(inv_y,inv_yhat)

RMSE = sqrt(mean_squared_error(inv_y, inv_yhat))

print('MSE: %.3f' % MSE + '   MAE: %.3f' % MAE + '   RMSE: %.3f' % RMSE)
def mean_absolute_percentage_error(y_true, y_pred): 

    """Calculates MAPE given y_true and y_pred"""

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



print(mean_absolute_percentage_error(inv_y,inv_yhat))
aa=[x for x in range(500)]

plt.figure(figsize=(8,4))

plt.plot(aa, inv_y[:500], marker='.', label="actual")

plt.plot(aa, inv_yhat[:500], 'r', label="prediction")



plt.tight_layout()

sns.despine(top=True)

plt.subplots_adjust(left=0.07)

plt.ylabel('PJME_MW', size=15)

plt.xlabel('Time step', size=15)

plt.legend(fontsize=15)

plt.show();