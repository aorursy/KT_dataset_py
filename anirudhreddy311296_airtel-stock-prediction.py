# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/airtelprice_data.csv')

len(data)
train_data = data[:int((len(data)+1)*.75)] 

test_data = data[int(len(data)*.75+1):] 

len(train_data), len(test_data)
train_data
trainset = train_data.iloc[:,1:2].values 
trainset
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_scaled = sc.fit_transform(trainset)
training_scaled
x_train = []

y_train = []



for i in range(60,3231):

    x_train.append(training_scaled[i-60:i,0])

    y_train.append(training_scaled[i,0])





x_train,y_train = np.array(x_train),np.array(y_train)
x_train.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
regressor = Sequential()

regressor.add(LSTM(units = 10,return_sequences = True,input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 10,return_sequences = True))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 10,return_sequences = True))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 10))

regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(x_train,y_train,epochs = 100, batch_size = 32)
test_data

real_stock_price = test_data.iloc[:,1:2].values
total = pd.concat((train_data['Open'],test_data['Open']),axis = 0)
inputs = total[len(total) - len(test_data)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
inputs.shape
x_test = []

for i in range(60,1076):

    x_test.append(inputs[i-60:i,0])

    
x_test = np.array(x_test)

x_test.shape
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

x_test.shape
predicted_price = regressor.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)
plt.plot(real_stock_price,color = 'red', label = 'Real Price')

plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')

plt.title('AIRTEL Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('AIRTEL Stock Price')

plt.legend()

plt.show()