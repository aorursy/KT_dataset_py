from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

from sklearn.preprocessing import MinMaxScaler

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
train = pd.read_csv('../input/trainset.csv')
test = pd.read_csv('../input/testset.csv')
metric = 'Open' # What matric to model and predict
timestamp = 64 # Maximum number of timestamps to learn from
layers = 3 # Number of layers in LSTM
neurons = 264 # Number of neurons in each layer
epochs = 64 # Number of times the entire data needs to be looped upon
batch_size = 16 # Weights updated after n rows
validation_split = 0.1 # Percentage of data to validate the model when training
dropout = 0.2 # Regularisation parameter
optimizer = 'adam' 
loss = 'mean_squared_error'
loc = train.columns.get_loc(metric)
train_data = train.iloc[:,loc:loc + 1].values
sc = MinMaxScaler(feature_range = (0,1))

train_data = sc.fit_transform(train_data)
train_x = []
train_y = []
for i in range(0, len(train_data) - timestamp - 1):
    train_x.append(train_data[i : i + timestamp, 0])
    train_y.append(train_data[i + timestamp, 0])
train_x, train_y = np.array(train_x), np.array(train_y)
train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))
regressor = Sequential()
regressor.add(LSTM(neurons, return_sequences = True, input_shape = (train_x.shape[1],1)))
regressor.add(Dropout(dropout))

if layers > 2:
    for i in range(2,layers):
        regressor.add(LSTM(neurons, return_sequences = True))
        regressor.add(Dropout(dropout))

regressor.add(LSTM(neurons))
regressor.add(Dropout(dropout))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = optimizer,loss = loss)
regressor.summary()
regressor.fit(train_x,train_y,epochs = epochs, batch_size = batch_size, validation_split = validation_split)
test_data = pd.concat((train[len(train) - timestamp:][metric], test[metric]), axis = 0)
test_data = test_data.values
test_data = test_data.reshape(-1,1)
test_data = sc.transform(test_data)
test_x = []
test_y = []
for i in range(0, len(test_data) - timestamp - 1):
    test_x.append(test_data[i : i + timestamp, 0])
    test_y.append(test_data[i + timestamp, 0])
test_x, test_y = np.array(test_x), np.array(test_y)
test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))
predicted_price = sc.inverse_transform(regressor.predict(test_x))
real_stock_price = test[metric]
def plot_pred(metric, real_stock_price, predicted_price):
    plt.plot(real_stock_price,color = 'red', label = 'Real Price')
    plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
    plt.title('Google ' +metric+ ' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google ' +metric+ ' Stock Price')
    plt.legend()
    plt.show()

plot_pred(metric, real_stock_price, predicted_price)