import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix



np.random.seed(7)



def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)



dataframe = pd.read_csv('../input/cpu-data-cleaned/CPU-Data.csv', usecols=[1])



dataset = dataframe.values

dataset = dataset.astype('float32')

# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]



look_back = 1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



# create and fit the LSTM network

model = Sequential()

model.add(LSTM(64,activation ='tanh', input_shape=(1, look_back), return_sequences=True))

model.add(LSTM(64, activation ='tanh', input_shape=(1, look_back), return_sequences=False))

#model.add(LSTM(32,activation ='tanh', input_shape=(1, look_back), return_sequences=False))

model.add(Dense(1, activation ='relu',))

model.compile(loss='mean_squared_error', optimizer='RMSProp', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=20, batch_size=32, verbose=2)

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

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
