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
def make_gru_network():

    regressorGRU = Sequential()

    # First GRU layer with Dropout regularisation

    

    regressorGRU.add(GRU(units = 30, return_sequences = False, input_shape=(1,2), activation='tanh'))

    regressorGRU.add(Dropout(0.3))

    

    # The output layer

    regressorGRU.add(Dense(units=1))

    return regressorGRU
model = make_gru_network()
model.compile(optimizer='rmsprop', loss='mean_squared_error')
x_train = np.array([1,2,2,3,3,4,4,5])

y_train = np.array([

    3,

    4,

    5,

    6

])
x_train = x_train.reshape(4,1,2)

x_train
model.fit(x_train, y_train, epochs = 30, batch_size = 2)
test_set = np.array([[[1,2]]])
model.predict(test_set)
# Load dataset from csv file

dataset = pd.read_csv('../historical_stock_prices.csv', index_col='date', parse_dates=['date'])
# Separate train/test

trainSet = dataset['2015':'2017'].sort_values(by=['ticker','date'])

testSet = dataset['2018':].sort_values(by=['ticker','date'])
trainSet.head()
# All symbols

symbols = trainSet.ticker.unique()
timesteps = 60

def make_samples(

    data_1,

    s):

    

    stop_append = {}

    for i in prediction_intervals:

        stop_append.update({i:False}) 

    l = len(data_1)

    

    for i in range(timesteps, l):

        x_1 = data_1[i-timesteps: i, 0]

        for j in stop_append:

            if not stop_append[j]:

                if i+j-1 < l:

                    y = data_1[i+j-1,0]

                    y = y.reshape(-1,1)

                    x_1 = x_1.reshape(-1,1)

                    

                    sc = MinMaxScaler(feature_range=(0,1))

                    sc.partial_fit(x_1)

                    sc.partial_fit(y)

                    

                    train_set[s]['x_1'][j].append(sc.transform(x_1))

                    train_set[s]['y'][j].append(sc.transform(y))

                    

                    if y == data_1[l-1,0]:

                        stop_append[j] =True
# Define train_set

train_set = {}

prediction_intervals = [1,3,5,10]

t = 1

for s in symbols:

    train_set.update({s:{

        'x_1':{},

        'y':{},

    }})

    for i in prediction_intervals:

        train_set[s]['x_1'][i] = []

        train_set[s]['y'][i] = []

            

    data_1 = trainSet.loc[trainSet['ticker'] == s][['adj_close']].values

        

    make_samples(

        data_1, 

        s)

    if t == 2000:

        break

    t += 1
# Define earlystopping callback function

es = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, mode='min', restore_best_weights=True)
def make_gru_network():

    regressorGRU = Sequential()

    # First GRU layer with Dropout regularisation

    

    regressorGRU.add(GRU(units = 30, return_sequences = False, input_shape=(1,60), activation='tanh'))

    regressorGRU.add(Dropout(0.3))

    

    # The output layer

    regressorGRU.add(Dense(units=1))

    return regressorGRU
# Training model with one feature in train_set

model = {}

model_history = {}

for i in prediction_intervals:

    model[i] = make_gru_network()

    model[i].compile(optimizer='rmsprop', loss='mean_squared_error')

    

    n_epoch = 100

    x_train = []

    y_train = []

    

    for s in train_set:

        for j in range (0, len(train_set[s]['x_1'][i])):

            x = [

                train_set[s]['x_1'][i][j],

            ]

            x_train.append(x)

        for j in train_set[s]['y'][i]:

            y_train.append(j)

    X_train, Y_train = np.array(x_train), np.array(y_train)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])

    Y_train = Y_train.reshape(Y_train.shape[0])

    

    print(X_train.shape, Y_train.shape)

    print('Fitting prediction interval {}  model'.format(i))

    model_history[i] = model[i].fit(X_train, Y_train, epochs = n_epoch, batch_size = 6000, 

                                    validation_split = 0.3,callbacks=[es])
for i in prediction_intervals:

    plt.plot(model_history[i].history['val_loss'])

    plt.plot(model_history[i].history['loss'])

    plt.title('Model Loss of Interval {}'.format(i))

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['val_loss', 'loss'], loc='upper right')

    plt.show()
# Calculate Mean squared_error

def return_mse(test,predicted):

    mse = mean_squared_error(test, predicted)

    return mse
# Plotting graph

def plotting_graph(y_test, y_pred, ticker, des):

    plt.plot(y_test, color='green', label='Actual adj_close value')

    plt.plot(y_pred, color='red', label='Predicted adj_close value')

    plt.title('Prediction of {} on {}'.format(ticker, des))

    plt.xlabel('Time steps')

    plt.ylabel('adj_close value')

    plt.legend()

    plt.show()
# Define test_set

def make_test_data(s):

    scaler = {s:{}}

    test_set = {s:{}}

    test_set.update({s:{

            'x_1':{},

            'y':{},

        }})

    for i in prediction_intervals:

            test_set[s]['x_1'][i] = []

            test_set[s]['y'][i] = []

            scaler[s][i] = []



    data_1 = testSet.loc[testSet['ticker'] == s][['adj_close']].values

    

    stop_append = {}

    for i in prediction_intervals:

        stop_append.update({i:False}) 

    l = len(data_1)



    for i in range(timesteps, l):

        x_1 = data_1[i-timesteps: i, 0]

        for j in stop_append:

            if not stop_append[j]:

                if i+j-1 < l:

                    y = data_1[i+j-1,0]

                    y = y.reshape(-1,1)

                    x_1 = x_1.reshape(-1,1)

                    

                    sc = MinMaxScaler(feature_range=(0,1))

                    sc.partial_fit(x_1)

                    sc.partial_fit(y)

                    scaler[s][j].append(sc)

                    

                    test_set[s]['x_1'][j].append(sc.transform(x_1))

                    test_set[s]['y'][j].append(y)

                    

                    if y == data_1[l-1,0]:

                        stop_append[j] =True

                    

    return test_set, scaler
def make_prediction(data, s, scaler):

    mse_val = {}

    y_true_val = {}

    y_pred_val = {}

    

    for i in prediction_intervals:

        x_test = []

        for j in range (0, len(data[s]['x_1'][i])):

            x = [

                data[s]['x_1'][i][j],

            ]

            x_test.append(x)



        X_test = np.array(x_test)

        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        

        result = model[i].predict(X_test)

        

        y_pred = []

        for j in range(0,len(result)):

            y = result[j]

            y = y.reshape(-1,1)

            k = scaler[s][i][j].inverse_transform(y)

            y_pred.append(k[0][0])

                    

        y_test = data[s]['y'][i]

        y_true = np.array(y_test)

        y_true = y_true.reshape(y_true.shape[0])

    

        y_true_val[i] = y_true

        y_pred_val[i] = y_pred



        mse_val[i] = return_mse(y_true, y_pred) 

    return y_true_val, y_pred_val, mse_val
s = 'AAPL'

data, sc = make_test_data(s)

y_true_val, y_pred_val, mse_val = make_prediction(data, s, sc)



for i in mse_val:

    print('Test Set -> MSE of {} inveral {}: {}'.format(s, i, mse_val[i]))
for i in prediction_intervals:

    plotting_graph(y_true_val[i], y_pred_val[i], s, 'Test Set interval {}'.format(i))
test_symbols = ['A', 'ACER', 'MSFT', 'ABC', 'AAPL']
all_mse = {}

for s in test_symbols:

    all_mse[s] = {}

    data, sc = make_test_data(s)

    y_true_val, y_pred_val, mse_val = make_prediction(data, s, sc)

    for i in prediction_intervals:

        all_mse[s][i] = mse_val[i]

        all_mse[s][i] = mse_val[i]
all_mse_1 = []

all_mse_3 = []

all_mse_5 = []

all_mse_10 = []

for s in all_mse:

    all_mse_1.append(round(all_mse[s][1], 2))

    all_mse_3.append(round(all_mse[s][3], 2))

    all_mse_5.append(round(all_mse[s][5], 2))

    all_mse_10.append(round(all_mse[s][10], 2))
# MSE of interval 1

all_mse_1
# MSE of interval 3

all_mse_3
# MSE of interval 5

all_mse_5
# MSE of interval 10

all_mse_10