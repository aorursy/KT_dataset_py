# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import SimpleRNN

from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/cleanThy.csv')

data.head()
#We use "Last Price" Column



data.columns = ["Date", "Last_Price", "Lowest_Price", "Highest_Price", "Volume"]

dataset = data.loc[:,["Last_Price"]].values

dataset = dataset.astype("float32")

dataset.shape
plt.figure(figsize=(30, 10))

sns.set(style="whitegrid")

sns.lineplot(data=dataset, palette="magma", linewidth=1.5, legend = False)

plt.show()
scaler = MinMaxScaler(feature_range=(0,1))

dataset = scaler.fit_transform(dataset)
#train-test split

train_size = int(len(dataset)/2)

test_size = len(dataset)-train_size

train = dataset[0:train_size,:]

test=dataset[train_size:len(dataset),:]

print("train size: {}, test size: {}".format(len(train), len(test)))
#train:



time_step = 10

datax=[]

datay=[]



for i in range(len(train)-time_step-1):

    a=train[i:(i+time_step),0]

    datax.append(a)

    datay.append(train[i+time_step,0])

trainx=np.array(datax)

trainy=np.array(datay)



#test:



datax=[]

datay=[]



for i in range(len(test)-time_step-1):

    a=test[i:(i+time_step),0]

    datax.append(a)

    datay.append(test[i+time_step,0])

testx=np.array(datax)

testy=np.array(datay)
trainx=np.reshape(trainx, (trainx.shape[0],1,trainx.shape[1]))

testx=np.reshape(testx, (testx.shape[0],1,testx.shape[1]))


model = Sequential()

model.add(LSTM(10, input_shape=(1, time_step))) #10 LSTM neuron

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainx, trainy, epochs=50, batch_size=1)
trainy = trainy.reshape(1,-1)

testy = testy.reshape(1,-1)
trainPredict = model.predict(trainx)

testPredict = model.predict(testx)

trainPredict = scaler.inverse_transform(trainPredict)

trainy = scaler.inverse_transform(trainy)

testPredict = scaler.inverse_transform(testPredict)

testy = scaler.inverse_transform(testy)
trainPredictPlot= np.empty_like(dataset)

trainPredictPlot[:,:]=np.nan

trainPredictPlot[time_step:len(trainPredict)+time_step, :] = trainPredict



testPredictPlot = np.empty_like(dataset)

testPredictPlot[:,:]=np.nan

testPredictPlot[len(trainPredict)+(time_step*2)+1:len(dataset)-1,:]=testPredict



plt.figure(figsize=(30, 10))

sns.set(style="whitegrid")

sns.lineplot(data=scaler.inverse_transform(dataset), palette="twilight", linewidth=1.5, legend = False)

sns.lineplot(data=trainPredictPlot, palette="BuPu", linewidth=1.0, legend = False)

sns.lineplot(data=testPredictPlot, palette="magma", linewidth=1.0, legend = False)

plt.show()


trainscore = math.sqrt(mean_squared_error(trainy[0], trainPredict[:,0]))

testscore = math.sqrt(mean_squared_error(testy[0], testPredict[:,0]))

print("Train MSE: {}, Test MSE: {}".format(trainscore, testscore))