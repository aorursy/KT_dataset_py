import numpy as np

import pandas as pd
from subprocess import check_output

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import GRU,LSTM

from keras.models import Sequential

from sklearn.model_selection import  train_test_split

import time

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from numpy import newaxis
ds =  pd.read_csv('../input/limited-nyse-price/prices-split-adjusted.csv' , header=0)

#Adjusted prices used for absolute consumer value

ds.head(5)
ms = ds[ds['symbol']=='MSFT']

ms_stock_prices = ms.close.values.astype('float32')
ms_stock_prices
ms_stock_prices.reshape(1762, 1)
ms_stock_prices.shape
plt.plot(ms_stock_prices)

plt.show()
sc = MinMaxScaler(feature_range=(0,1))

ms_dataset = sc.fit_transform(ms_stock_prices.reshape(-1,1))
ms_dataset.shape
train_size = int(0.80 * len(ms_dataset))

test_size = len(ms_dataset)-train_size
train , test = ms_dataset[0:train_size,:], ms_dataset[train_size:len(ms_dataset),:]

print(len(train))

print(len(test))
#lookback is Number of steps to check for output

def create_dataset(dataset, look_back=5):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)
#since singular data 

look_back = 5

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
trainX.shape

sv_trainX = trainX

sv_trainY = trainY

sv_testX = testX

sv_testY = testY
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

trainX.shape
trainX.shape

trainX
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
print(testX.shape)

print(testY.reshape(-1,1))
recc_model = Sequential()
#RNN Model

recc_model.add(GRU(input_shape=(trainX.shape[1],1), units=100, return_sequences=True))

#recc_model.add(Activation('relu'))

recc_model.add(Dropout(0.2))

#hidden_1

recc_model.add(GRU(units=50,return_sequences=True))

#recc_model.add(Activation('relu'))

recc_model.add(Dropout(0.2))

#hidden_2

#recc_model.add(Dense(units=50, activation = 'relu'))

#recc_model.add(GRU(units=50,return_sequences=True))

#recc_model.add(Dropout(0.2))

#Fourth Hidden with no return sequences

recc_model.add(GRU(units=50))

#recc_model.add(Dropout(0.2))

#output Layer

recc_model.add(Dense(units=1))

#recc_model.add(Activation('linear'))
recc_model.summary()

recc_model.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])
history = recc_model.fit(

    trainX,

    trainY,

    batch_size=64,

    epochs=30,

    validation_split=0.05)
recc_model_lstm = Sequential()

#RNN Model

recc_model_lstm.add(LSTM(input_shape=(trainX.shape[1],1), units=100, return_sequences=True))

recc_model_lstm.add(Activation('relu'))

recc_model_lstm.add(Dropout(0.2))

#hidden_1_lstm

recc_model_lstm.add(LSTM(units=50,return_sequences=False))

recc_model_lstm.add(Activation('relu'))

recc_model_lstm.add(Dropout(0.2))

#hidden_2

#recc_model.add(Dense(units=50, activation = 'relu'))

#recc_model_lstm.add(LSTM(units=50,return_sequences=True))

#recc_model_lstm.add(Activation('relu'))

#recc_model_lstm.add(Dropout(0.2))

#Fourth Hi_lstmdden with no return sequences

#recc_model_lstm.add(Dropout(0.2))

#output La_lstmyer

recc_model_lstm.add(Dense(units=1))

#recc_model_lstm.add(Activation('linear'))

recc_model_lstm.summary()

recc_model_lstm.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])

history_lstm = recc_model_lstm.fit(

    trainX,

    trainY,

    batch_size=64,

    epochs=30,

    validation_split=0.05)
from sklearn.svm import SVR

clf = SVR(C=1.0, epsilon=0.2)

clf.fit(sv_trainX, sv_trainY)

print ("SVR Score =" , clf.score(sv_trainX, sv_trainY, sample_weight=None))

pred = clf.predict(sv_testX)

pred = pred.reshape(347,1)

pred.reshape(1,-1)

from sklearn.neighbors import KNeighborsRegressor

neigh = KNeighborsRegressor(n_neighbors=5)

neigh.fit(sv_trainX, sv_trainY)

print ("KNN Score =" , neigh.score(sv_trainX, sv_trainY, sample_weight=None))

pred_KNN = neigh.predict(sv_testX)

pred_KNN = pred_KNN.reshape(347,1)
plt.figure(figsize=(10, 6), dpi=100)

plt.plot(history.history['loss'], label='GRU train', color='brown')

plt.plot(history.history['val_loss'], label='GRU test', color='blue')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()

plt.title('Training and Validation loss')

plt.show()

predicted_stock_price = recc_model.predict(testX)

predicted_lstm = recc_model_lstm.predict(testX)

print(predicted_stock_price)
predicted_stock_price.shape

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

predicted_lstm = sc.inverse_transform(predicted_lstm)

pred_sv = sc.inverse_transform(pred)

pred_KNN_l = sc.inverse_transform(pred_KNN)

real_prices = ms_dataset[train_size:]

real_prices = sc.inverse_transform(real_prices)

real_prices

import math
plt.plot(real_prices, color = 'red',label = 'Real Prices')

plt.plot(predicted_stock_price, color = 'yellow', label = 'GRU')

plt.plot(predicted_lstm, color = 'green', label = 'LSTM')

plt.plot(pred_sv, color= 'black', label = 'Support Regressor')

plt.plot(pred_KNN_l, color = 'blue', label = 'KNN')

plt.title('Price Prediction [Reduced]')

plt.legend()

plt.show()
train_acc = recc_model.evaluate(trainX, trainY, verbose=0)

test_acc = recc_model.evaluate(testX, testY, verbose=0)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()