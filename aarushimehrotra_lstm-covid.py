# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# load data into numpy array
raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

# separate data by country
cnames = set()
snames = set()
data = dict()
for i in raw.to_numpy():
    if type(i[1]) == float:
        cnames.add(i[2])
    else:
        snames.add(i[1])
        
for cname in cnames:
    current = raw[raw["Country_Region"] == cname].to_numpy()
    data.update({cname: current})
for sname in snames:
    current = raw[raw["Province_State"] == sname].to_numpy()
    data.update({sname: current})
    
# x and y values
x_data = list(range(len(data[list(data.keys())[0]])))
y_data = dict()
for (name, cases) in data.items():
    y_data.update({name : cases[:,4]})
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

name = "California"

def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create differenced series
def difference(data, interval=1):
    diff = list()
    for i in range(interval, len(data)):
        value = data[i] - data[i-interval]
        diff.append(value)
    return diff

# scales data to [-1, 1]
def scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data)
    data = data.reshape(data.shape[0], data.shape[1])
    data_scaled = scaler.transform(data)
    return scaler, data_scaled

# data pre-processing
differenced = difference(y_data[name], 1)
supervised = timeseries_to_supervised(differenced, 1)
supervised = supervised.values
scaler, scaled = scale(supervised)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def fit_lstm(data, batch_size, epochs, neurons):
    x, y = data[:, 0:-1], data[:, -1]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(epochs):
        model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forecast(model, batch_size, x):
    #x = row[0:-1]
    x = x.reshape(1, 1, len(x))
    yhat = model.predict(x, batch_size=batch_size)
    return yhat[0, 0]
lstm_model = fit_lstm(scaled, 1, 3000, 4)
reshaped = scaled[:, 0].reshape(len(scaled), 1, 1)
lstm_model.predict(reshaped, batch_size=1)
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# invert differenced series
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# inverts scale for given value
def inverse_scale(scaler, data, value):
    new_row = [i for i in data] + [value]
    arr = np.array(new_row)
    arr = arr.reshape(1, len(arr))
    inverted = scaler.inverse_transform(arr)
    return inverted[0, -1]

predictions = list()
for i in range(150):
    x, y = scaled[i, 0:-1], scaled[i, -1]
    yhat = forecast(lstm_model, 1, x)
    yhat = inverse_scale(scaler, x, yhat)
    yhat = inverse_difference(y_data[name], yhat, 150+1-i)
    predictions.append(yhat)
    expected = y_data[name][i + 1]
    print('Day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

rmse = (mean_squared_error(y_data[name][1:], predictions)) ** 0.5
print('RMSE: %.3f' % rmse)
        
plt.plot(y_data[name], label="actual")
plt.plot(predictions, label="model")
plt.legend()
plt.show()