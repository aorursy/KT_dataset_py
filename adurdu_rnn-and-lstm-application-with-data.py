# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

#import libraries and packages:

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv('../input/google-stock-price/Google_Stock_Price_Train.csv')
dataset_train.head()
#open'ı kullanıcaz sadece:

train = dataset_train.loc[:, ['Open']].values #array'e çevirdik

train
from sklearn.preprocessing import MinMaxScaler #bununla, 0-1 arasına scale ettik

scaler = MinMaxScaler(feature_range = (0, 1))

train_scaled = scaler.fit_transform(train)

train_scaled
plt.plot(train_scaled)
#ilk 1-50 yi alıp X_train'e, 51. data point'i de y_train'e,

#2-51'i alıp X_train'e, 52'yi y_train'e ...olacak şekilde data frame i oluşturuyoruz:

X_train = []

y_train = []

timesteps = 50



for i in range(timesteps, 1250):

    X_train.append(train_scaled[i - timesteps:i, 0])

    y_train.append(train_scaled[i, 0])

    

X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping:

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train.shape
y_train.shape
# ***********************************

# Model of Recurrent Neural Network

# RNN

# ***********************************



#Initialize RNN:

regressor = Sequential()



#Adding the first RNN layer and some Dropout regularization

regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True, input_shape= (X_train.shape[1],1)))

regressor.add(Dropout(0.2))



#Adding the second RNN layer and some Dropout regularization

regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))

regressor.add(Dropout(0.2))



#Adding the third RNN layer and some Dropout regularization

regressor.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))

regressor.add(Dropout(0.2))



#Adding the fourth RNN layer and some Dropout regularization

regressor.add(SimpleRNN(units = 50))

regressor.add(Dropout(0.2))



#Adding the output layer

regressor.add(Dense(units = 1))



#Compile the RNN

regressor.compile(optimizer='adam', loss='mean_squared_error')



#Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# ***********************************

# Model of Long Short Term Memory (LSTM)

# LSTM model

# ***********************************

regressor1 = Sequential()

regressor1.add(LSTM(10, input_shape= (X_train.shape[1],1))) # 10 lstm neuron(block)

regressor1.add(Dense(1))

regressor1.compile(loss='mean_squared_error', optimizer='adam')

regressor1.fit(X_train, y_train, epochs=30, batch_size=1)
dataset_test = pd.read_csv('../input/google-stock-pricetest/Google_Stock_Price_Test.csv')

dataset_test.head()
real_stock_price = dataset_test.loc[:, ['Open']].values

#real_stock_price
#Getting the predicted stock price

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

inputs = dataset_total[len(dataset_total)-len(dataset_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs) #minmax scaler

#inputs
X_test = []

y_test = []

for i in range(timesteps, 70):

    X_test.append(inputs[i-timesteps:i,0])

    y_test.append(inputs[i, 0])

X_test = np.array(X_test)

y_test = np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price1 = regressor1.predict(X_test)

predicted_stock_price_tr = regressor.predict(X_train)

predicted_stock_price1_tr = regressor1.predict(X_train)

predicted_stock_price_RNN = scaler.inverse_transform(predicted_stock_price)

predicted_stock_price_LSTM = scaler.inverse_transform(predicted_stock_price1)

predicted_stock_price_RNN_tr = scaler.inverse_transform(predicted_stock_price_tr)

predicted_stock_price_LSTM_tr = scaler.inverse_transform(predicted_stock_price1_tr)

#inverse_transform ile, scale edildikten sonra predict edilen değerleri gerçek değer aralığına çekiyoruz 
y_train1 = scaler.inverse_transform([y_train])

y_test1 = scaler.inverse_transform([y_test])

# calculate root mean squared error for RNN

trainScore_RNN = math.sqrt(mean_squared_error(y_train1[0], predicted_stock_price_RNN_tr[:,0]))

print('Train Score of RNN: %.2f RMSE' % (trainScore_RNN))

testScore_RNN = math.sqrt(mean_squared_error(y_test1[0], predicted_stock_price_RNN[:,0]))

print('Test Score of RNN: %.2f RMSE' % (testScore_RNN))

# calculate root mean squared error for LSTM

trainScore_LSTM = math.sqrt(mean_squared_error(y_train1[0], predicted_stock_price_LSTM_tr[:,0]))

print('Train Score of LSTM: %.2f RMSE' % (trainScore_LSTM))

testScore_LSTM = math.sqrt(mean_squared_error(y_test1[0], predicted_stock_price_LSTM[:,0]))

print('Test Score of LSTM: %.2f RMSE' % (testScore_LSTM))



plt.plot(scaler.inverse_transform(y_train.reshape(-1,1)), color='red', label='Real Google Stock Price')

plt.plot(predicted_stock_price_RNN_tr, color='blue', label='Predicted Google Stock Price with RNN')

plt.plot(predicted_stock_price_LSTM_tr, color='black', label='Predicted Google Stock Price with LSTM')

plt.title('Google Stock Price Prediction for Training')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.subplots_adjust(bottom=0.5, top=1)

plt.show()





plt.plot(real_stock_price, color='red', label='Real Google Stock Price')

plt.plot(predicted_stock_price_RNN, color='blue', label='Predicted Google Stock Price with RNN')

plt.plot(predicted_stock_price_LSTM, color='black', label='Predicted Google Stock Price with LSTM')

plt.title('Google Stock Price Prediction for Testing')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()



plt.subplots_adjust(bottom=0.5, top=1)

plt.show()
data = pd.read_csv('../input/international-airline-passengers/international-airline-passengers.csv')

data.head()
dataset = data.iloc[:, 1].values

plt.plot(dataset)

plt.xlabel('time')

plt.ylabel('number of passengers (in thousands)')

plt.title('Passengers')

plt.show()
dataset = dataset.reshape(-1,1) #(145, ) iken (145,1)e çevirdik

dataset = dataset.astype('float32')

dataset.shape
scaler = MinMaxScaler(feature_range= (0,1))

dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset)*0.5)

test_size = len(dataset)- train_size



train = dataset[0:train_size, :]

test = dataset[train_size:len(dataset), :]



print('train size: {}, test size: {}'.format(len(train), len(test)))
dataX = []

datay = []

timestemp = 10



for i in range(len(train)- timestemp -1):

    a = train[i:(i+timestemp), 0]

    dataX.append(a)

    datay.append(train[i + timestemp, 0])



    

trainX, trainy = np.array(dataX), np.array(datay)
dataX = []

datay = []

for i in range(len(test)- timestemp -1):

    a = test[i:(i+timestemp), 0]

    dataX.append(a)

    datay.append(test[i + timestemp, 0])



    

testX, testy = np.array(dataX), np.array(datay)
trainX.shape
trainX = np.reshape(trainX, (trainX.shape[0],1,  trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0],1,  testX.shape[1]))
trainX.shape
# ***********************************

# Model of Long Short Term Memory (LSTM)

# LSTM model

# ***********************************



model = Sequential()

model.add(LSTM(10, input_shape=(1, timestemp))) # 10 lstm neuron(block)

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainy, epochs=100, batch_size=1)
# ***********************************

# Model of Recurrent Neural Network

# RNN

# ***********************************



#Initialize RNN:

model1 = Sequential()



#Adding the first RNN layer and some Dropout regularization

model1.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True, input_shape=(1, timestemp)))

model1.add(Dropout(0.2))



#Adding the second RNN layer and some Dropout regularization

model1.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))

model1.add(Dropout(0.2))



#Adding the third RNN layer and some Dropout regularization

model1.add(SimpleRNN(units = 50, activation='tanh', return_sequences=True))

model1.add(Dropout(0.2))



#Adding the fourth RNN layer and some Dropout regularization

model1.add(SimpleRNN(units = 50))

model1.add(Dropout(0.2))



#Adding the output layer

model1.add(Dense(units = 1))



#Compile the RNN

model1.compile(optimizer='adam', loss='mean_squared_error')



#Fitting the RNN to the Training set

model1.fit(trainX, trainy, epochs=200, batch_size=32)
#make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

trainPredict1 = model1.predict(trainX)

testPredict1 = model1.predict(testX)
# invert predictions

trainy1 = scaler.inverse_transform([trainy])

testy1 = scaler.inverse_transform([testy])

trainPredict_RNN = scaler.inverse_transform(trainPredict1)

testPredict_RNN = scaler.inverse_transform(testPredict1)

trainPredict_LSTM = scaler.inverse_transform(trainPredict)

testPredict_LSTM = scaler.inverse_transform(testPredict)
# calculate root mean squared error

trainScore_RNN = math.sqrt(mean_squared_error(trainy1[0], trainPredict_RNN[:,0]))

print('Train Score: %.2f RMSE for RNN' % (trainScore_RNN))

testScore_RNN = math.sqrt(mean_squared_error(testy1[0], testPredict_RNN[:,0]))

print('Test Score: %.2f RMSE for RNN' % (testScore_RNN))

trainScore_LSTM = math.sqrt(mean_squared_error(trainy1[0], trainPredict_LSTM[:,0]))

print('Train Score: %.2f RMSE for LSTM' % (trainScore_LSTM))

testScore_LSTM = math.sqrt(mean_squared_error(testy1[0], testPredict_LSTM[:,0]))

print('Test Score: %.2f RMSE for LSTM' % (testScore_LSTM))


plt.plot(trainy1.T, color='red', label='Real number of passengers')

plt.plot(trainPredict_RNN, color='blue', label='Predicted number of passengers with RNN')

plt.plot(trainPredict_LSTM, color='black', label='Predicted number of passengers with LSTM')

plt.title('Passengers for Training')

plt.xlabel('Time')

plt.ylabel('number of passengers (in thousands)')

plt.legend()

plt.subplots_adjust(bottom=0.5, top=1)

plt.show()





plt.plot(testy1.T, color='red', label='Real number of passengers')

plt.plot(testPredict_RNN, color='blue', label='Predicted number of passengers with RNN')

plt.plot(testPredict_LSTM, color='black', label='Predicted number of passengers with LSTM')

plt.title('Passengers for Testing')

plt.xlabel('Time')

plt.ylabel('number of passengers (in thousands)')

plt.legend()



plt.subplots_adjust(bottom=0.5, top=1)

plt.show()

# shifting train

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[timestemp:len(trainPredict)+timestemp, :] = trainPredict

# shifting test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(timestemp*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()