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

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/5) Recurrent Neural Network/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
train_data.head()
train = train_data.loc[:,['Open']].values

#values using for converting array
train
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

train_scaled = scaler.fit_transform(train)

train_scaled
plt.plot(train_scaled)
# Creating a data structure with 50 timesteps and 1 output

X_train = []

y_train = []

timesteps = 50

for i in range(timesteps, 1258):

    X_train.append(train_scaled[i-timesteps:i, 0])

    y_train.append(train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
y_train
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

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
test_data = pd.read_csv('/kaggle/input/5) Recurrent Neural Network/Stock_Price_Test.csv')
test_data.head()
real_stock_price = test_data.loc[:,['Open']].values
real_stock_price
total_data = pd.concat((train_data['Open'],test_data['Open']),axis=0)

inputs = total_data[len(total_data)-len(test_data)-timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs) #min max scaler
inputs
X_test = []

for i in range(timesteps, 70):

    X_test.append(inputs[i-timesteps:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')

plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')

plt.title('Google Stoc Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()