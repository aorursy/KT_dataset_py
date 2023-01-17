# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Data Creation

path = "../input/price-volume-data-for-all-us-stocks-etfs/Stocks/googl.us.txt"

file = open(path)

columns = list(file.readline().strip().split(","))



data = []

for line in file.readlines():

    data.append(line.strip().split(","))



df = pd.DataFrame(data=data, columns=columns)

df.head()
# Dropping unnecessary columns

data = df.reset_index()["Close"].astype("float32")

data.head()
# Visualizing the closing price

try:

    sns.lineplot(data=data[:, 0], label="Closing Price")

except:

    sns.lineplot(data=data, label="Closing Price")
# Scaling

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

data=scaler.fit_transform(np.array(data).reshape(-1,1))

print(data.shape)
# Splitting dataset into train and test split

train_size = int(len(data) * 0.65)

test_size = len(data) - train_size

train_data, test_data = data[0:train_size,:], data[train_size:len(data), :1]

print(train_size, test_size)
# Convert an array of values into a dataset matrix

def create_dataset(dataset, time_step=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-time_step-1):

        a = dataset[i:(i+time_step), 0]    

        dataX.append(a)

        dataY.append(dataset[i + time_step, 0])

    return np.array(dataX), np.array(dataY)
# Preparing the dataset

time_step = 90

X_train, y_train = create_dataset(train_data, time_step)

X_test, y_test = create_dataset(test_data, time_step)
# X and y shapes

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)
# Reshape input to be [samples, time steps, features] which is required for LSTM

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] , 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] , 1)

print(X_train.shape, X_test.shape)
# Libraries

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM
# Model

model=Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))

model.add(LSTM(32))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
# Run the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=32, batch_size=32)
# Predicting the results

train_predict = model.predict(X_train)

test_predict = model.predict(X_test)

results = np.concatenate((train_predict, test_predict), axis=0)
# Original vs Predicted

try:

    sns.lineplot(data=data[:, 0], label="Original")

    sns.lineplot(data=results[:, 0], label="Predicted")

except:

    sns.lineplot(data=data, label="Original")

    sns.lineplot(data=results, label="Predicted")