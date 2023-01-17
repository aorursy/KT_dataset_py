# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import libraries

import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error
dataset = pd.read_csv('../input/WEATHERDATA_2016-2018.csv', encoding='unicode_escape')

dataset.head()
# check for missing values

dataset.columns[dataset.isnull().any()].tolist()
dataset.columns
dataset.dtypes
# # filter data to only get a specified year

# year_interested = 2018



# df_mask = dataset['Year']==year_interested

# df = dataset[df_mask].copy()



# df.head()



# if i am interested in all year

df = dataset
# convert year, month and day to datetime

df['Date'] = pd.to_datetime(df.Year*10000+df.Month*100+df.Day,format='%Y%m%d')

df.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)
df.head()
# only take the columns that are considered

df = df[['Date', 'Mean Temperature (°C)']]
df.plot(x='Date', y=['Mean Temperature (°C)'], kind='line', figsize=(16,4))
df.set_index('Date', inplace=True)

df.head()
# split data into training and testing sets

train_set = df[:'2018-5'].values

test_set = df['2018-6':].values
# scaling the dataset

sc = MinMaxScaler(feature_range=(0,1))

train_set_scaled = sc.fit_transform(train_set)

test_set_scaled = sc.transform(test_set)



train_set_scaled.shape
# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output

# So for each element of training set, we have 30 previous training set elements 

num_storage = 10



X_train = []

y_train = []

for i in range(num_storage, train_set_scaled.shape[0]):

    X_train.append(train_set_scaled[i-num_storage:i,0])

    y_train.append(train_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)



X_test = []

y_test = []

for i in range(num_storage, test_set_scaled.shape[0]):

    X_test.append(test_set_scaled[i-num_storage:i,0])

    y_test.append(test_set[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_train.shape
# reshape X_train for efficient modelling

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))



X_train.shape
# The LSTM architecture

regressor = Sequential()

# First LSTM layer with Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))

regressor.add(Dropout(0.2))

# Second LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Third LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Fourth LSTM layer

regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))

# The output layer

regressor.add(Dense(units=1))



# Compiling the RNN

regressor.compile(optimizer='rmsprop',loss='mean_squared_error')

# Fitting to the training set

regressor.fit(X_train,y_train,epochs=50,batch_size=32)
predictions = regressor.predict(X_test)

predictions = sc.inverse_transform(predictions)
def plot_predictions(test, predicted):

    plt.plot(test, color='red', label='Real Values')

    plt.plot(predicted, color='blue', label='Predicted Values')

    plt.title('Singapore 2018 Weather Prediction')

    plt.xlabel('Date')

    plt.ylabel('Mean Temperature (°C)')

    plt.show()
# plot results for LSTM

plot_predictions(y_test, predictions)
def return_rmse(test, predicted):

    rmse = math.sqrt(mean_squared_error(test, predicted))

    print("the root mean squared error is {}".format(rmse))
# evaluate model

return_rmse(y_test, predictions)