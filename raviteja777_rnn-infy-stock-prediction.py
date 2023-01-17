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
#importing libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#import data from csv



stock_data = pd.read_csv('/kaggle/input/infy-stock-data/INFY.csv')

stock_data.head()
stock_open = stock_data.iloc[:,1].values



stock_open
stock_open.shape
#split the data into train and test

#use first 4000 rows for training and rest of them for test



train_limit = 4000



stock_train = stock_open[0:train_limit]

stock_test = stock_open[train_limit:]



#visualizing first five values of train and test

print("----Train -----")

print(*stock_train[:5])

print("----Test -----")

print(*stock_test[:5])
print(stock_train.shape)

print(stock_test.shape)
#Scaling the data set 

#using Min Max Scalar

from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler(feature_range=(0,1))

stock_train_scaled = sc.fit_transform(stock_train.reshape(-1,1))



stock_train_scaled[:5]
#taking the time steps

time_steps = 60

X_train = []

y_train = []



for i in range(time_steps,train_limit):

    X_train.append(stock_train_scaled[i-time_steps:i,0])

    y_train.append(stock_train_scaled[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)
#input shape for LSTM shud be 3 dimensional

# (samples,time steps ,features )

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))



X_train[:5]
from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Dropout
regressor = Sequential()



#add LSTM layer

regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (X_train.shape[1],1)))

regressor.add(Dropout(0.2))
# add another LSTM layer

regressor.add(LSTM(units = 50,return_sequences = True))

regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
#compiling the neural network 

regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
#training the data



regressor.fit(X_train,y_train,epochs = 50, batch_size = 32)
# scaling the test data 



stock_test_scaled = sc.transform(stock_test.reshape(-1,1))



stock_test_scaled.shape
stock_test_scaled
X_test = []

for i in range(time_steps,stock_test_scaled.shape[0]):

    X_test.append(stock_test_scaled[i-time_steps:i,0])



X_test = np.array(X_test)



X_test[:5]
X_test.shape
#transform test data same way as train 



X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

X_test.shape
real_stock_price = stock_test[time_steps:]

real_stock_price
predicted_stock_price = regressor.predict(X_test)



#reverse scaling predicted price 

predicted_stock_price = sc.inverse_transform(predicted_stock_price).flatten()



predicted_stock_price
#plotting real_stock_price in red vs predicted_stock_price in blue



plt.plot(real_stock_price,color = 'red', label = 'Real Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Price')

plt.title('Infy Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Infy Stock Price')

plt.legend()

plt.show()