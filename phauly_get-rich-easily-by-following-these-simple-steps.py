#Load libs

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential

import matplotlib.pyplot as plt
#Lets load the daily prices



btc = pd.read_csv('../input/bitcoin_price.csv')

eth = pd.read_csv('../input/ethereum_price.csv')

btc.head()
#data is in reverse order

btc = btc.iloc[::-1]

eth = eth.iloc[::-1]

btc.head()
#getting the 4 price-related features from the dataframe

features = btc[["Open","High","Low","Close"]].values

print(features.shape)



#we change the data to have something more generalizeable, lets say [ %variation , %high, %low]

price_variation = (1- (features[:,0]/features[:,3]))*100

highs = (features[:,1]/np.maximum(features[:,0],features[:,3]) -1)*100

lows = (features[:,2]/np.minimum(features[:,0],features[:,3]) -1)*100



X_train = np.array([price_variation]).transpose()

#little trick to make X_train a 3 dimensional array for LSTM input shape

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))



print(X_train[:2])



#We generate Y_train. For this update, we will only determine if the trend is up or down for 2 days ahead

Y_train = np.array((np.sign((features[2:,3]/features[:-2,3]-1))+1)/2)

print(Y_train[:10])
#Lets make a simple lstm model 

#I got it from online tutorial

model = Sequential()

model.add(LSTM(100,

               input_shape = (None,1),

               return_sequences = True

              ))

model.add(Dropout(0.1))

model.add(LSTM(100, return_sequences=True))

model.add(Dropout(0.1))

model.add(LSTM(50))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss="mse", optimizer="rmsprop")
#train the model

model.fit(X_train[:-2],Y_train, batch_size=512,

	    epochs=500,

	    validation_split=0.05)

features_eth = eth[["Open","High","Low","Close"]].values

features_eth = features_eth[0:8]

print(features_eth.shape)

print(features_eth)

print(features_eth[:,0])

print(features_eth[:,3])

print('here')

print(features_eth[2:,3])

print(features_eth[-2:,3])

variations = (features_eth[:,3]-features_eth[:,0])

#print(variations)



#we change the data to have something more generalizeable, lets say [ %variation , %high, %low]

price_variation = (1- (features_eth[:,0]/features_eth[:,3]))*100

highs = (features_eth[:,1]/np.maximum(features_eth[:,0],features_eth[:,3]) -1)*100

lows = (features_eth[:,2]/np.minimum(features_eth[:,0],features_eth[:,3]) -1)*100



X_test = np.array([variations]).transpose()

print(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

print(X_test)

Y_test = np.array((np.sign(features_eth[2:,3]/features_eth[:-2,3]-1)+1)/2)

print(Y_test)
model.evaluate(X_test[:-2],Y_test)
pred = model.predict(X_test)



predicted = (np.sign(pred-0.45)+1)/2*50
#lets plot the last predictions in comparison to the actual variations

start =650

stop = 700

plt.plot(predicted[start:stop],'r')#prediction is in red.

plt.plot(features_eth[start:stop,3],'b')#actual in blue.

plt.plot(Y_test[start:stop]*50)

plt.show()
predicted