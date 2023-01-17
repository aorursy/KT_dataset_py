# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Importing the training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output

# We use 60 because  60 days back and forward to predect ,and 1258 the no of train set range no

# 0 is the coulmn index

X_train = []

y_train = []

for i in range(60, 1258):

    X_train.append(training_set_scaled[i-60:i, 0])

    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping

# we have to reshape the data into 3D

# 1s is no of lines in X_train ,2nd is no of times step(coulmn od xtrain),3rd is no of predector the open stock prise

# for mode details go to keras recurent layer docomentation

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# Initialising the RNN

regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation

# We use return_sequence true because we are addind anather layer after it, at the last layer it will be False



regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation

regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))
# Adding the output layer

# For Full connection layer we use dense

# As the output is 1D so we use unit=1

regressor.add(Dense(units = 1))
# Compiling the RNN

# For optimizer we can go through keras optimizers Docomentation

# As it is regression problem so we use mean squared error

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set

# For best fit accourding to data we can increase the epochs

# For forward & back propageted and update weights we use 32  stock prises to train 

regressor.fit(X_train, y_train, epochs = 150, batch_size = 20)
# Getting the real stock price of 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):

    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()