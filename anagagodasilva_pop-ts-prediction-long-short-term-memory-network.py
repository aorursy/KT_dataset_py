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
import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
df_POP = pd.read_csv('/kaggle/input/population-time-series-data/POP.csv', delimiter=',')

df_POP.dataframeName = 'POP.csv'

print(df_POP.shape)
df_POP.head(5)
df_POP.tail(5)
plt.title('Monthly Population TS')

plt.plot(df_POP['value'])

plt.show()
df_POP['date']=pd.to_datetime(df_POP['date'])
plt.title('Monthly Population TS')

plt.plot(df_POP['date'],df_POP['value'])

plt.show()
# random seed for reproducibility

np.random.seed(5)
# reshape the dataset

dataset=np.array(df_POP['value'])

dataset=dataset.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.67)

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

	return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1

look_back = 1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
# reshape input to [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions

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
# shift train predictions for plotting

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(df_POP['date'],scaler.inverse_transform(dataset),color='blue')

plt.plot(df_POP['date'],trainPredictPlot,color='green')

plt.plot(df_POP['date'],testPredictPlot,color='red')

plt.show()
# plot baseline and predictions between 1994 & 2010

plt.plot(df_POP['date'][500:700],(scaler.inverse_transform(dataset))[500:700],color='blue')

plt.plot(df_POP['date'][500:700],trainPredictPlot[500:700],color='green')

plt.plot(df_POP['date'][500:700],testPredictPlot[500:700],color='red')

plt.show()