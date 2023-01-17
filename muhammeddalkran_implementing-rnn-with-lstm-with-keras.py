# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/international-airline-passengers/international-airline-passengers.csv', skipfooter = 5,engine='python')

data.head()
dataset = data.iloc[:,1].values

plt.plot(dataset)

plt.xlabel('time')

plt.ylabel('# of Passenger')

plt.title('International Airline Passenger')

plt.show()
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
dataset = dataset.reshape(-1,1)

dataset = dataset.astype("float32")

dataset.shape
scaler = MinMaxScaler(feature_range = (0,1))

dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.50)

test_size = len(dataset) - train_size

train = dataset[0:train_size,:]

test = dataset[train_size:len(dataset),:]

print("Train size : {}".format(len(train)))

print("Test size : {}".format(len(test)))
time_stemp = 10

dataX = []

dataY = []

for i in range(len(train) - time_stemp - 1):

    a = train[i:(i+time_stemp),0]

    dataX.append(a)

    dataY.append(train[i+time_stemp,0])

trainX = np.array(dataX)

trainY = np.array(dataY)


dataX = []

dataY = []

for i in range(len(train) - time_stemp - 1):

    a = test[i:(i+time_stemp),0]

    dataX.append(a)

    dataY.append(test[i+time_stemp,0])

testX = np.array(dataX)

testY = np.array(dataY)
trainX = np.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))

testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

model = Sequential()

model.add(LSTM(10,input_shape=(1,time_stemp))) #10LSTM neuron(block)

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(trainX,trainY,epochs = 50, batch_size = 1)
train_predict = model.predict(trainX)

test_predict = model.predict(testX)

train_predict = scaler.inverse_transform(train_predict)

trainY = scaler.inverse_transform([trainY])

test_predict = scaler.inverse_transform(test_predict)

testY= scaler.inverse_transform([testY])

train_score = math.sqrt(mean_squared_error(trainY[0],train_predict[:,0]))

print('Train Score %.2f RMSE' %(train_score))

test_score = math.sqrt(mean_squared_error(testY[0],test_predict[:,0]))

print('Test Score %.2f RMSE' %(test_score))
train_predict_plot = np.empty_like(dataset)

train_predict_plot[:,:] = np.nan

train_predict_plot[time_stemp:len(train_predict) +time_stemp,:]=train_predict

test_predict_plot = np.empty_like(dataset)

test_predict_plot[:,:] = np.nan

test_predict_plot[len(train_predict)+(time_stemp*2)+1:len(dataset) -1,:] = test_predict

plt.plot(scaler.inverse_transform(dataset))

plt.plot(train_predict_plot)

plt.plot(test_predict_plot)

plt.show()
