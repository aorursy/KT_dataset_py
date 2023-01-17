import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualisation

import math # mathematical functions
df = pd.read_csv('../input/apple-stocks/datasets_8388_11883_AAPL_2006-01-01_to_2018-01-01.csv')
df.head()
df.tail()
plt.figure(figsize = (8,5), dpi = 200)

plt.plot(df['Close'])

plt.xlabel('Days')

plt.ylabel('Stock Price')

plt.show()
df.info()
train_set = df.iloc[:3005,4:5].values

print("Training Set:")

print(train_set)

test_set = df.iloc[3005:3019,4:5].values

print("Test Set")

print(test_set)
#Data Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

train_set = scaler.fit_transform(train_set)

train_set
# Using an LSTM timestep of 60, i.e, we'll be looking 2 months back.

X_train = []

Y_train = []

for i in range(60,3005):

    X_train.append(train_set[i-60:i,0])

    Y_train.append(train_set[i,0])

X_train = np.array(X_train)

Y_train = np.array(Y_train)
# Reshaping

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
from keras.models import Sequential

from keras.layers import LSTM, Dropout, Dense





# LSTM model with 4 layers and dropout = 0.2 for all layers

model = Sequential()





model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))

model.add(Dropout(0.2))



model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))



model.add(LSTM(units = 50, return_sequences = True))

model.add(Dropout(0.2))



model.add(LSTM(units = 50))

model.add(Dropout(0.2))



# Output layer

model.add(Dense(units = 1))
model.summary()
# Compilation

model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
# Fitting the model to the training set

model.fit(X_train, Y_train, epochs = 150, batch_size = 32)
# Manipulating the small test set to look back by 60 days.

X_test_inputs = df.iloc[len(df) - len(test_set) - 60:,4:5].values

X_test_inputs = X_test_inputs.reshape(-1,1)

X_test_inputs = scaler.transform(X_test_inputs)

X_test_inputs.shape
X_test = []

for i in range(60,74):

    X_test.append(X_test_inputs[i-60:i,0])

X_test = np.array(X_test)
# Reshaping X_test

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
# Predicting the stock prices

stock_pred = model.predict(X_test)

stock_pred = scaler.inverse_transform(stock_pred)
from sklearn.metrics import mean_squared_error

error = math.sqrt(mean_squared_error(test_set,stock_pred))

print(error)
plt.figure(figsize = (8,5), dpi = 200)

plt.plot(test_set, color = 'red', label = 'True Stock Price')

plt.plot(stock_pred, color = 'green', label = 'Predicted Stock Price')

plt.title('Apple Stock Prediction')

plt.xlabel('Days')

plt.ylabel('Stock Price')

plt.legend()

plt.show()