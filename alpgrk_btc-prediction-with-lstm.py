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
import pandas as pd

import numpy as np

#import requests

#import json

import datetime

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.layers import LSTM, Bidirectional, TimeDistributed

from keras import losses

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error

import math

import matplotlib.pyplot as plt
# Load Dataset



#url = "https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=1&aggregate=1&allData=true"

#response = requests.get(url)

#json_data = response.json()

#df = pd.DataFrame(json_data["Data"])



df = pd.read_csv("../input/btc-daily-price-all-data/BTC Daily Price(All Data).csv")

df.drop(["Unnamed: 0"],axis = 1, inplace = True)

df["time"] = [datetime.datetime.fromtimestamp(i) for i in df["time"]]

df.drop(index=(len(df) - 1), inplace=True)
df
df.describe()
df.info()
# visualization

plt.plot(df["close"])

plt.xlabel("Time")

plt.ylabel("Close Prices")

plt.show()
# reshape dataset



data = df["close"].values

data = data.reshape(-1,1)

data = data.astype("float32")
# Normalization



scaler = MinMaxScaler()

data = scaler.fit_transform(data)
# Train Test Split



train_size = int(len(data) * 0.80)

test_size = int(len(data)) - train_size

train = data[0:train_size, :]

test = data[train_size:len(data), :]
# Train Dataset

time_step = 30

dataX = []

dataY = []



for i in range(len(train) - time_step - 1):

    a = train[i:(i + time_step), 0]

    dataX.append(a)

    dataY.append(train[i + time_step, 0])



trainX = np.array(dataX)

trainY = np.array(dataY)



# Test Dataset



dataX = []

dataY = []



for i in range(len(test) - time_step - 1):

    a = test[i:(i + time_step), 0]

    dataX.append(a)

    dataY.append(test[i + time_step, 0])



testX = np.array(dataX)

testY = np.array(dataY)
# Reshape Test and Train Dataset



trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# %% Model Create



model = Sequential()

model.add(LSTM(100, activation="tanh", input_shape=(1, time_step)))  # return_sequences=True for add another LSTM layer

# model.add(LSTM(100, activation="tanh"))

model.add(Dropout(0.5))

# model.add(LSTM(60, activation="tanh"))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

history = model.fit(trainX, trainY, epochs=6, validation_split=0.33, validation_data=(testX, testY))
# %% Prediction and Visualization



trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))

print('Train Score: %.2f RMSE' % trainScore)

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

print('Test Score: %.2f RMSE' % testScore)



# shifting train

trainPredictPlot = np.empty_like(data)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[time_step:len(trainPredict) + time_step, :] = trainPredict



# shifting test predictions for plotting

testPredictPlot = np.empty_like(data)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict) + (time_step * 2) + 1:len(data) - 1, :] = testPredict



# plot baseline and predictions

plt.figure(figsize=(12, 8))

plt.plot(scaler.inverse_transform(data))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
# Statistic



plt.plot(history.history["loss"], color="blue", label="Train")

plt.plot(history.history["val_loss"], color="red", label="Test")

plt.legend()

plt.show()
# Prediction



# Load Dataset



#url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=32"

#response = requests.get(url)

#json_data_predict = response.json()

#df_predict = pd.DataFrame(json_data_predict["Data"]["Data"])



df_predict = pd.read_csv("../input/btc-daily-pricefor-prediction/BTC Daily Price(for prediction).csv")



df_predict.drop(["conversionType","conversionSymbol","Unnamed: 0"],axis = 1, inplace=True)



df_predict["time"] = [datetime.datetime.fromtimestamp(i) for i in df_predict["time"]]

df_predict.drop(index=(len(df_predict) - 1), inplace=True)





# normalization and reshape

data_predict = df_predict["close"].values

data_predict = data_predict.reshape(-1, 1)

data_predict = data_predict.astype("float32")

data_predict = scaler.fit_transform(data_predict)



# Predict Dataset



time_step = 30

dataX = []

dataY = []



for i in range(len(data_predict) - time_step - 1):

    a = data_predict[i:(i + time_step), 0]

    dataX.append(a)

# Reshape



predict_X = np.array(dataX)

predict_X = np.reshape(predict_X, (predict_X.shape[0], 1, predict_X.shape[1]))
# Make a Prediction



y_pred = model.predict(predict_X)

y_pred = scaler.inverse_transform(y_pred)

print(y_pred)