# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)





# Any results you write to the current directory are saved as output.
# Importing the libraries



data = pd.read_csv("/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/trn.us.txt")
data.head()
traindata = data.loc[:,["Open"]].values
traindata.shape
# split into train and test sets



train_size = int(len(traindata) * 0.9938)

test_size = len(traindata) - train_size

train, test = traindata[0:train_size,:], traindata[train_size:len(traindata),:]

print(len(train), len(test))
train.shape
test.shape
# Feature Scaling



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range= (0,1))

train_scaled =scaler.fit_transform(train)

train_scaled
plt.plot(train_scaled)

plt.show()
# Creating a data structure with 50 timesteps and 1 output



X_train = []

y_train = []

timesteps = 50

print(train_scaled[0, 0])

for i in range(timesteps, train.shape[0]):

   

    X_train.append(train_scaled[i-timesteps:i, 0])

    y_train.append(train_scaled[i, 0])

    

X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train.shape


# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout



# Initialising the RNN

regressor = Sequential()



# Adding the first RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))



# Adding a second RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a fourth RNN layer and some Dropout regularisation

regressor.add(SimpleRNN(units = 50))

regressor.add(Dropout(0.2))

# Adding the output layer

regressor.add(Dense(units = 1))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 250, batch_size = 32)
# Getting the predicted stock price of 2017

dataset_total = data['Open']

inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)

inputs

inputs.shape
# min max scaler

inputs = scaler.transform(inputs)  

X_test = []

for i in range(timesteps, 70):

    X_test.append(inputs[i-timesteps:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



# Visualising the results



plt.plot(test, color = 'red', label = 'Real  Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted  Stock Price')

plt.title('Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel(' Stock Price')

plt.legend()

plt.show()

data.columns
dataset = data.iloc[:,1].values

plt.plot(dataset)

plt.xlabel("Time")

plt.ylabel("Stock Price")

plt.title("Stock Market Dataset")

plt.show()
dataset = dataset.reshape(-1,1)

dataset = dataset.astype("float32")

dataset.shape
# scaling 

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.50)

test_size = len(dataset) - train_size

train = dataset[0:train_size,:]

test = dataset[train_size:len(dataset),:]

print("train size: {}, test size: {} ".format(len(train), len(test)))
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


import matplotlib.pyplot as plt

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
# model

model = Sequential()

model.add(LSTM(10, input_shape=(1, time_stemp))) # 10 lstm neuron(block)

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=250, batch_size=1)
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

plt.plot(scaler.inverse_transform(dataset[:200]))

plt.plot(trainPredictPlot[:200])

plt.plot(testPredictPlot[:200])

plt.show()