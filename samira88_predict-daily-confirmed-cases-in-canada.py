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

import requests
import numpy as np
import pandas as pd
import io
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
 

#Load Time Series Data


filename = ("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")
data = pd.read_csv(filename)
data.head()


Canada_confirmed = data.loc[data['Country/Region'] == 'Canada']
Canada_confirmed 

# Visualizing a time series

from matplotlib import pyplot
pyplot.figure(figsize=(20,10)) 

for r in Canada_confirmed['Province/State']:  
        pyplot.plot(range(len(Canada_confirmed.columns)-4), Canada_confirmed.loc[Canada_confirmed['Province/State']==r].iloc[0,4:], label = r) 
        
         
pyplot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
pyplot.title('Total Number of COVID-19 Confirmed Cases in pronvices of Canada')
pyplot.xlabel('Day')
pyplot.ylabel('Number of Cases')
# Canada_TS is the summation of all pronvises confirmed cases

from pandas import DataFrame
Canada_TS=0

for i in range(len (Canada_confirmed['Province/State'])):
     Canada_TS = Canada_confirmed.iloc[i,4:]+ Canada_TS
        
print (Canada_TS.values)
 # fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset

dataframe=np.resize(Canada_TS,(len(Canada_TS),1))
dataset = dataframe.astype('float32')
print(dataframe.shape)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset.shape)

# split into train and test sets
train_size = (len(dataset) - 14)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


trainY[0]
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
#calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# plot

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
 
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions


plt.figure(2, figsize=(12,9))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot, marker='.')
plt.plot(testPredictPlot, marker='.')
plt.xlabel('Day Number', fontsize=14)
plt.ylabel('Number of Confirmed cases in Canada',fontsize=14)
plt.legend()
plt.title('Daily Prediction of COVID-19 Time confirmed cases in Canada ', fontsize=20)

plt.show()
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(np.resize(Canada_TS,(len(Canada_TS),1)))
#print(dataset1)
#Canada_TS.values.shape

#raw_seq =dataset1
#l=len(raw_seq)

 

# define input sequence


dataset1 = dataset[0:len(dataset),0]

raw_seq =dataset1
l=len(raw_seq)

# demonstrate prediction
#x_input = array([raw_seq[l-3], raw_seq[l-2], raw_seq[l-1]])
x_input = np.array([raw_seq[l-1]])
x_input = x_input.reshape((1, 1, 1))
yhat = model.predict(x_input)
trainPredict1 = scaler.inverse_transform(yhat)
print(trainPredict1)
#window=np.zeros((3,),np.float32)
#print(l)
for i in range(l-1,l+5):
    x_input = np.array([raw_seq[i]])
    x_input = x_input.reshape((1, 1, 1))
    yhat = model.predict(x_input)
    trainPredict1 = scaler.inverse_transform(yhat)
    #print(i,len(raw_seq))
    if i+1==len(raw_seq):
        raw_seq=np.append(raw_seq,yhat)
    else:
        raw_seq[i+1]=yhat
    print(i+1,'day '+str(i-l+2),trainPredict1[0]) 
quick 