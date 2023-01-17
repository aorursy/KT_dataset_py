# RNN

import pandas as pd

import random

import os

import copy

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__) 

import cufflinks as cf

# For Notebooks

init_notebook_mode(connected=True)

# For offline use

cf.go_offline()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/aaxn.us.txt')

dataset.head()
# Creating an numpy arrray of dataset

training_set = dataset.iloc[:,0:2]
training_set
training_set.iplot(kind='line',y='Open',x='Date')
training_set = dataset.iloc[2000:-20,0:2]
training_set.iplot(kind='line',y='Open',x='Date')
# feature scaling

training_set = dataset.iloc[2000:-20,1:2].values

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

training_set = mms.fit_transform(training_set)
X_train = []

y_train = []



# creating a time series:     use timesteps --> 100 

time_step = 100;

for i in range(time_step,len(training_set)):

    X_train.append(training_set[i-time_step:i,0])

    y_train.append(training_set[i,0])



#converting list into array

X_train = np.array(X_train);y_train = np.array(y_train)



# We know that LSTM layer takes 3 dimentional array

#The LSTM input layer must be 3D.

#The meaning of the 3 input dimensions are: samples, time steps, and features.

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

X_train.shape
#Building an rnn



#importing the keras libraries

from keras.models import Sequential

from keras.layers import Dropout,Dense,LSTM

#adding layers

regressor = Sequential()

regressor.add(LSTM(units=200,return_sequences = True,input_shape=(X_train.shape[1],1)))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units=200,return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units=200,return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units=200,return_sequences = True))

regressor.add(Dropout(0.2))

regressor.add(LSTM(units=100))

regressor.add(Dropout(0.2))



#output layer

regressor.add(Dense(units=1))



regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(X_train,y_train,epochs=100,batch_size=80)
losses = regressor.history.history

losses = pd.DataFrame(losses)

losses['loss'].plot()
dataset_test  = dataset.iloc[-20:,1:2]
real_stocks = dataset_test['Open'].values
dataset_total =dataset.iloc[:,1:2].values
dataset_total = mms.transform(dataset_total)
time_step=100

prediction_stocks = []

# creating a time series:     use timesteps --> 80 

for i in range(len(dataset_total)-20,len(dataset_total)):

    prediction_stocks.append(dataset_total[i-time_step:i])
prediction_stocks = np.array(prediction_stocks)
prediction_stocks = np.reshape(prediction_stocks,(prediction_stocks.shape[0],prediction_stocks.shape[1],1))

predictions = regressor.predict(prediction_stocks)

predictions = mms.inverse_transform(predictions)
#visulising the results



plt.plot(real_stocks, color = 'red', label = 'Real Stock Price')

plt.plot(predictions, color = 'blue', label = 'Predicted Stock Price')

plt.title('Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()
R = pd.DataFrame(real_stocks)

P = pd.DataFrame(predictions)

Data = pd.concat([R,P],axis=1)
Data.columns=['RealStocks','PredictedStocks']
Data[['RealStocks','PredictedStocks']].iplot(kind='spread')