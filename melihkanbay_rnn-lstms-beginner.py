# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/international-airline-passengers/international-airline-passengers.csv",skipfooter = 5)

data.info()
import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
dataset = data.iloc[:,1].values

plt.plot(dataset)

plt.show()
dataset = dataset.reshape(-1,1)

dataset = dataset.astype("float32")

dataset.shape
scaler = MinMaxScaler()

dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset)*0.50)

test_size = len(dataset) - train_size

train = dataset[0:train_size,:]

test = dataset[train_size:len(dataset),:]

print(len(train),len(test))
time_stemp = 10

dataX = []

dataY = []

for i in range(len(train)-time_stemp-1):

    a = train[i:(i+time_stemp), 0]

    dataX.append(a)

    dataY.append(train[i + time_stemp, 0])

trainX = np.array(dataX)

trainY = np.array(dataY)  
dataX = []

dataY = []

for i in range(len(test)-time_stemp-1):

    a = test[i:(i+time_stemp), 0]

    dataX.append(a)

    dataY.append(test[i + time_stemp, 0])

testX = np.array(dataX)

testY = np.array(dataY)  
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()

model.add(LSTM(15, input_shape=(1,time_stemp)))

model.add(Dense(1))

model.compile(loss = "mean_squared_error",optimizer = "adam")

model.fit(trainX, trainY, epochs = 50, batch_size =2)
trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shifting train

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict

# shifting test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.figure(figsize = (12,8))

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.savefig(fname = "final")

plt.legend()

plt.show()