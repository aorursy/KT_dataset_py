import numpy as np

import pandas as pd

import os

path = "../input/climate-change-earth-surface-temperature-data"

lokasi = os.listdir(path)

print(lokasi)
data = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByMajorCity.csv")

data.head()

#uniqueValues = data.nunique()

#print(uniqueValues)
hmm = data['City'].values.ravel()

cariunik = pd.unique(hmm)

print(cariunik)
data = data[data['City']=='Surabaya']

data.head()
data.describe()
data.info()
import matplotlib.pyplot as plt



data[['AverageTemperature']].plot() 

plt.show()
data.hist(bins=20)

plt.show()
data.size
data['dt'] = pd.to_datetime(data['dt'])

#data[] = pd.to_datetime('13000101', format='%Y%m%d'

data = data.set_index('dt')

data.head(30)
data.tail(30)
data.size
data.isna().sum()
data[['AverageTemperature']].plot()

plt.show()
date_split = '1869-01-01'

data = data[date_split:]

data.head()
print(data.info())

print(data.describe())
data.isna().sum()
data_inter = data.interpolate()

print(data_inter.info())

print(data_inter.isna().sum())



data_inter[['AverageTemperature']].plot()

plt.show()
data = data_inter['AverageTemperature'].values

#data = data.reshape(1, -1)

print(data.shape)



from sklearn.preprocessing import MinMaxScaler



data = data.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

print(scaled_data)

scaled_data_series = scaled_data.reshape(1,-1)

print(scaled_data_series)
plt.plot(scaled_data)

plt.show()
def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)



# from --> https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
raw_seq = scaled_data_series[0]



# choose a number of time steps

n_steps = 12



# split into samples

X, y = split_sequence(raw_seq, n_steps)



# summarize the data

for i in range(len(X)):

    print(X[i], y[i])



print(X.shape)

print(X[1])
train_data = X[:1500]

test_data = X[1500:]



train_y = y[:1500]

test_y = y[1500:]



y1 = np.arange(1,train_data.shape[0]+1)

y2 = np.arange(train_data.shape[0],train_data.shape[0]+test_data.shape[0])



plt.plot(y1, train_data, 'r', y2, test_data, 'b')

plt.show()
from keras.models import Sequential

from keras.layers import Activation, Dense

from keras.layers import LSTM

from keras.layers import Dropout

from keras import  layers

import keras
n_features = 1

train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], n_features)

test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], n_features)
train_data.shape
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(train_data, train_y, epochs=100, batch_size=2, verbose=2)
loss = model.evaluate(train_data, train_y, verbose=0)

print(loss)
predictions = model.predict(test_data, verbose=2)

predictions = predictions.reshape(predictions.shape[0])

print(predictions.shape, test_y.shape)

fig = plt.figure(figsize=(16, 7))

plt.plot(y2, predictions, 'r', y2, test_y, 'b')

plt.show()
regressor = Sequential()



regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (train_data.shape[1], 1)))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))



regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



regressor.add(Dense(units = 1))



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



regressor.fit(train_data, train_y, epochs = 100, batch_size = 4)
predictions2 = regressor.predict(test_data, verbose=2)

predictions2 = predictions2.reshape(predictions.shape[0])
fig = plt.figure(figsize=(16, 7))

plt.plot(y2, predictions2, 'r', y2, test_y, 'b')

plt.show()