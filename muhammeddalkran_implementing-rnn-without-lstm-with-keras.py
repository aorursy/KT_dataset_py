# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings 

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset_train = pd.read_csv('/kaggle/input/Stock_Price_Train.csv')

dataset_train.head()
train = dataset_train.loc[:,["Open"]].values

train
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

train_scaled = scaler.fit_transform(train)

train_scaled

plt.plot(train_scaled)

plt.show()
X_train = []

y_train = []

timesteps = 50

for i in range(timesteps, 1258):

    X_train.append(train_scaled[i-timesteps:i, 0])

    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
y_train
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import SimpleRNN

from keras.layers import Dropout

regressor = Sequential()
regressor.add(SimpleRNN(units = 50, activation = 'tanh',return_sequences = True, input_shape = (X_train.shape[1],1)))

regressor.add(Dropout(0.2))



regressor.add(SimpleRNN(units = 50, activation = 'tanh',return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(SimpleRNN(units = 50, activation = 'tanh',return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(SimpleRNN(units = 50))

regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
regressor.fit(X_train,y_train,epochs = 100, batch_size = 32)
test_dataset = pd.read_csv('/kaggle/input/Stock_Price_Test.csv')

test_dataset.head()
real_stock_price = test_dataset.loc[:,['Open']].values

real_stock_price
dataset_total = pd.concat((dataset_train['Open'],test_dataset['Open']),axis = 0)

inputs = dataset_total[len(dataset_total) - len(test_dataset) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)

inputs
X_test = []

y_test = []

for i in range(timesteps, 70):

    X_test.append(inputs[i-timesteps:i,0])

    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
import math

from sklearn.metrics import mean_squared_error



train_predict = regressor.predict(X_train)

test_predict = regressor.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)

trainY = scaler.inverse_transform([y_train])

test_predict = scaler.inverse_transform(test_predict)

testY= scaler.inverse_transform([y_test])

train_score = math.sqrt(mean_squared_error(trainY[0],train_predict[:,0]))

print('Train Score %.2f RMSE' %(train_score))

test_score = math.sqrt(mean_squared_error(testY[0],test_predict[:,0]))

print('Test Score %.2f RMSE' %(test_score))
plt.plot(real_stock_price, color =  'red', label = 'real_stock_price')

plt.plot(predicted_stock_price, color = 'blue', label = 'predicted_stock_price without lstm' )

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show