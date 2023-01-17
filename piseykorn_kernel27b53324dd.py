import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Dropout, GRU, Bidirectional, LSTM

from keras.optimizers import SGD

import math

from sklearn.metrics import mean_squared_error

import keras
# Load dataset

dataset = pd.read_csv('/kaggle/input/daily-historical-stock-prices-1970-2018/historical_stock_prices.csv', index_col='date', parse_dates=['date'])
# Separate train and test data

train_data = dataset[:'2017'].sort_values(by=['ticker','date'])

test_data = dataset['2018-01-01':'2018-06-30'].sort_values(by=['ticker','date'])
timesteps = 60
# All symbols

symbols = dataset.ticker.unique()
def make_train_set(data,s):

    l = len(data)

    for i in range(timesteps, l):

        x = data[i-timesteps: i, 0]

        y = data[i,0]

        

        x = x.reshape(-1,1)

        y = y.reshape(-1,1)

        

        sc = MinMaxScaler(feature_range=(0,1))

        sc.partial_fit(x)

        sc.partial_fit(y)



        train_set[s]['x'].append(sc.transform(x))

        train_set[s]['y'].append(sc.transform(y))
# Define train_set

train_set = {}

for s in ['A', 'ACER', 'MSFT', 'ABC', 'AAPL']:  

    train_set.update({s:{'x':[],'y':[]}})

    data = train_data.loc[train_data['ticker'] == s][['adj_close']].values

    

    make_train_set(data,s)
# Define earlystopping callback function

es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, mode='min', restore_best_weights=True)
def make_gru_network():

    regressorGRU = Sequential()

    # First GRU layer with Dropout regularisation

    

    regressorGRU.add(GRU(units = 30, return_sequences = False, input_shape=(60,1), activation='tanh'))

    regressorGRU.add(Dropout(0.3))

    

    # The output layer

    regressorGRU.add(Dense(units=1))

    return regressorGRU
# Training multi symbols model

model = make_gru_network()

model.compile(optimizer='rmsprop', loss='mean_squared_error')



for s in train_set:

    print ("Training {}".format(s))

    X_train, Y_train = np.array(train_set[s]['x']), np.array(train_set[s]['y'])

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

    Y_train = Y_train.reshape(Y_train.shape[0])



    print(X_train.shape, Y_train.shape)

    model_history= model.fit(X_train, Y_train, epochs = 20, batch_size = 60, 

                                    validation_split = 0.3,callbacks=[es])
# Plotting validation loss and loss

plt.plot(model_history.history['val_loss'])

plt.plot(model_history.history['loss'])

plt.ylabel('loss')

plt.xlabel('epoch')

plt.title('Loss graph of {}'.format(s))

plt.show()
# Test a particular symbol by using Test data

s = "AAPL"

test_adj_close_values = test_data.loc[test_data['ticker'] == s][['adj_close']].values
def make_test_set(s, data):

    l = len(data)

    scaler = []

    test_set = {'x':[], 'y':[]}

    y_true = []

    

    for i in range(timesteps, l):

        x = data[i-timesteps: i, 0]

        y = data[i,0]

        y_true.append(y)



        x = x.reshape(-1,1)

        y = y.reshape(-1,1)



        sc = MinMaxScaler(feature_range=(0,1))

        sc.partial_fit(x)

        sc.partial_fit(y)



        scaler.append(sc)



        test_set['x'].append(sc.transform(x))

        test_set['y'].append(sc.transform(y))

        

    return test_set, y_true, scaler
def make_prediction(s, data):

    test_set, y_true, scaler = make_test_set(s, data)

    X_test = np.array(test_set['x'])

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    result = model.predict(X_test)

    y_pred = []

    for i in range(0,len(result)):

        y = result[i]

        y = y.reshape(-1,1)

        k = scaler[i].inverse_transform(y)

        y_pred.append(k[0][0])

    return y_true, y_pred
y_true, y_pred = make_prediction(s, test_adj_close_values)
# Plotting actual vs predicted values

plt.plot(y_true, color='green', label='Actual value')

plt.plot(y_pred, color='red', label='Predicted value')

plt.xlabel('Time steps')

plt.ylabel('Adj close value')

plt.title('Adj close values of {}'.format(s))

plt.legend()

plt.show()
# Take 60 real values

test_data_2 = test_adj_close_values[0:60]
# Make prediction by using predicted values or combination of real values and predicted values

def make_prediction_2(test_data, all_data):

    test_data.reshape(-1,1)

    sc = MinMaxScaler(feature_range=(0,1))

    X_test = np.array(sc.fit_transform(test_data))

    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])

    result = model.predict(X_test)

    for i in range(0,len(result)):

        y = result[i]

        y = y.reshape(-1,1)

        k = sc.inverse_transform(y)

        y_pred = k[0][0]

        

    all_data.append(y_pred)

    if len(all_data) == len(test_adj_close_values) - timesteps:

        return all_data

    else:

        new_test_data = np.append(np.delete(test_data,0), [y_pred]).reshape(-1,1)

        return make_prediction_2(new_test_data, all_data)
y_pred_2 = make_prediction_2(test_data_2, [])
# Plotting graph

plt.plot(y_pred, color='red', label='Predicted value method 1')

plt.plot(y_pred_2, color='blue', label='Predicted value method 2')

plt.plot(y_true, color='green', label='Actual value')

plt.xlabel('Time steps')

plt.ylabel('adj_close value')

plt.title('Ajusted values of {}'.format(s))

plt.legend()

plt.show()