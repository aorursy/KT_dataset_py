import numpy as np
import pandas as pd
import datetime
def parser(x):
    return datetime.datetime.strptime('190'+x, '%Y-%m')

# set "squeeze=True" to return a Series instead of a DataFrame
series = pd.read_csv('../input/shampoo.csv', parse_dates=[0], date_parser=parser, index_col=0, squeeze=True)
series
series.plot()
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

raw_values = series.values
diff_values = difference(raw_values, 1)
diff_values
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

supervised = timeseries_to_supervised(diff_values, 1)
supervised
supervised_values = supervised.values
train, test = supervised_values[0:-12], supervised_values[-12:]
train
test
from sklearn.preprocessing import MinMaxScaler

def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    #train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    #test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

scaler, train_scaled, test_scaled = scale(train, test)
train_scaled
test_scaled
from keras.models import Sequential
from keras.layers import Dense, LSTM

def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])  # i.e. (n_sample, n_time_step, n_feature)
    model = Sequential()
    # State in the LSTM layer between batches is cleared by default, therefore we must make the LSTM stateful. 
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    # Add a single neuron in the output layer with a linear activation to predict the sales at the next time step.
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # By default, the samples within an epoch are shuffled prior to being exposed to the network.
        # Since we want the network to build up state as it learns across the sequence of observations,
        # we can disable the shuffling of samples by setting “shuffle” to “False“.
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        # Reset the internal state at the end of the training epoch, ready for the next training iteration.
        model.reset_states()
    return model
# The batch_size must be set to 1, because it must be a factor of the size of the training and test datasets.
lstm_model = fit_lstm(train_scaled, batch_size=1, nb_epoch=3000, neurons=4)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
# The batch_size must be set to 1 because we are interested in making one-step forecasts on the test data.
# Seed the initial state by making a prediction on all samples in the training dataset. In theory, the internal state should be set up ready to forecast the next time step.
lstm_model.predict(train_reshaped, batch_size=1)
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))  # len(X): n_feature
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
from math import sqrt
from sklearn.metrics import mean_squared_error

rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
from matplotlib import pyplot

pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)