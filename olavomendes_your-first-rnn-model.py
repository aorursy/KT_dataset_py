import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



from tensorflow import random

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout
dataset_training = pd.read_csv('../input/amazonstockprice/AMZN_train.csv')

dataset_training.head()
training_data = dataset_training[['Open']].values

training_data
scaler = MinMaxScaler(feature_range=(0, 1))



# Preprocessing training data

training_data_scaled = scaler.fit_transform(training_data)

training_data_scaled
X_train = []

y_train = []



# 60 timestamps

for i in range(60, 1258):

    X_train.append(training_data_scaled[i-60:i, 0])

    y_train.append(training_data_scaled[i, 0])

    

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
seed = 1

np.random.seed(seed)

random.set_seed(seed)
model = Sequential()



# First LSTM layer

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))



# Second LSTM layer

model.add(LSTM(50, return_sequences=True))



# Third LSTM layer

model.add(LSTM(50, return_sequences=True))



# Fourth LSTM layer

model.add(LSTM(50))



# Output layer

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)
dataset_testing = pd.read_csv('../input/amazonstockprice/AMZN_test.csv')

actual_stock_price = dataset_testing[['Open']].values

actual_stock_price
total_data = pd.concat((dataset_training['Open'], dataset_testing['Open']), axis=0)
inputs = total_data[len(total_data) - len(dataset_testing) - 60:].values

inputs = inputs.reshape(-1, 1)

inputs = scaler.transform(inputs)
X_test = []



for i in range(60, 81):

    X_test.append(inputs[i-60:i, 0])



X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)



# Reescale the data

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
# Actual price

plt.plot(actual_stock_price, color = 'blue',label = 'Real Amazon Stock Price', ls='--')

# Predicted price

plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Amazon Stock Price', ls='-')



plt.title('PREDICTED AMAZON STOCK PRICE')

plt.xlabel('Time in days')

plt.ylabel('Real Stock Price')

plt.legend()



plt.show()