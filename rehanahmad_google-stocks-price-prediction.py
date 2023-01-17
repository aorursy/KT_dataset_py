# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/trainset.csv")

df_train.head()
# Lets work on the open stock price only

training_data = df_train.iloc[:,1:2]

print(training_data.shape)

plt.plot(training_data)

plt.ylabel('Price')
# Normalize the training data between [0,1]

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(0,1))

training_data_scaled = mms.fit_transform(training_data)

plt.plot(training_data_scaled)
# Create trainind data x_train with window/history of 60 samples and y_train with one future sample. 

x_train=[]

y_train=[]

for i in range(60,training_data.shape[0]):

    x_train.append(training_data_scaled[i-60:i,0])

    y_train.append(training_data_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

print('x_train shape: ',x_train.shape)

print('y_train shape: ',y_train.shape)
from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout



x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



# Create model using LSTM, Dropout and Dense layer as an output layer. 

net = Sequential()

net.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))

net.add(Dropout(0.2))

net.add(LSTM(units=50, return_sequences=True))

net.add(Dropout(0.2))

net.add(LSTM(units=50))

net.add(Dropout(0.2))

net.add(Dense(units=1))

net.compile(optimizer='adam', loss='mean_squared_error')
# Train the model

history = net.fit(x_train, y_train, epochs=100, batch_size=32)
plt.plot(history.history['loss'])
test_data = pd.read_csv("../input/testset.csv")

real_sotck_values = test_data.iloc[:,1:2].values

dataset_total = pd.concat((training_data['Open'], test_data['Open']), axis=0, ignore_index=True)

print(dataset_total.shape)

plt.plot(dataset_total)
inputs = dataset_total[len(dataset_total) - len(test_data)-60:].values

print(inputs.shape)

plt.plot(inputs)

inputs=inputs.reshape(-1,1)
inputs = mms.transform(inputs)

plt.plot(inputs)

x_test = []

for i in range(60, inputs.shape[0]):

    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

print(x_test.shape)
predicted_price = net.predict(x_test)

predicted_price = mms.inverse_transform(predicted_price)

plt.plot(predicted_price)
plt.plot(real_sotck_values, color='red', label='Real price')

plt.plot(predicted_price, color='blue', label='Predicted price')

plt.title('Google stock price prediction')

plt.xlabel('Date/Time')

plt.ylabel('Google stock price')

plt.legend()

plt.show()