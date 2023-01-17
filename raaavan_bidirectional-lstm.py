# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, Dense,Dropout,LSTM,Activation, RepeatVector, SimpleRNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def read_data(data, index_col = 0):
    data = pd.read_csv(filepath)
    return data

filepath = "../input/hourly-energy-consumption/AEP_hourly.csv"
df = read_data(filepath, index_col=0)
df.head()
# convert dates into datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.head()

# df['hour'] = df['Datetime'].dt.year
# extract date related features

def data_conversion(data, index='none'):
    data = df.copy()
    
    data['year'] = data['Datetime'].dt.year
    data['month'] = data['Datetime'].dt.month
    data['day'] = data['Datetime'].dt.day
    data['week'] = data['Datetime'].dt.dayofweek
    data['yearday'] = data['Datetime'].dt.dayofyear
    return data

data = data_conversion(df)
data.info()
data.loc[data['year'] == 2018]
df[:107401]
# check for null values

df.isnull().sum()
# perform mean min-max normalization 

dataset = df.AEP_MW.values.reshape(-1,1)

sclar = MinMaxScaler(feature_range=(0,1))
dataset = sclar.fit_transform(dataset)
dataset.shape
def load_data(dataset, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(dataset)):
        X_train.append(dataset[i - seq_len: i, 0])
        y_train.append(dataset[i, 0])

    # last year i.e. 2018 data is used for testing
    X_test = X_train[107401:]
    y_test = y_train[107401:]

    # remaining data is used for training.
    X_train = X_train[:107401]
    y_train = y_train[:107401]

    # convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return [X_train, y_train, X_test, y_test]
seq_len = 20 #choose sequence length

X_train, y_train, X_test, y_test = load_data(dataset, seq_len)

X_train = X_train.reshape(X_train.shape[0], seq_len, 1)
X_test = X_test.reshape(X_test.shape[0], seq_len, 1)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(LSTM(40, return_sequences=True, input_shape=(X_train.shape[1],1))))
model.add(Bidirectional(LSTM(20,return_sequences= False, activation= 'linear')))
model.add(tf.keras.layers.Dropout(0.5))

model.add(Dense(1))
# model.add(layers.LSTM(256))

model.compile(loss='MSE', optimizer = 'adam')
hist = model.fit(X_train, y_train, epochs=10, validation_split=0.3, batch_size=1000)
model.summary()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(hist.history['loss'], 'b' ,label = 'train loss', linewidth=2)
ax.plot(hist.history['val_loss'], 'r', label ='Validation loss', linewidth=2)
ax.set_title('model loss')
ax.set_ylabel('mse')
ax.set_xlabel('epoch')
ax.legend()
plt.show()

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
print("mean sequare error = ",mse)
def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16, 4))
    plt.plot(test, color='blue', label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='red', label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()


plot_predictions(y_test, pred, "Predictions made by model")
seasons = np.array(['Winter', 'Spring', 'Summer', 'Fall', 'Winter'])
f = np.searchsorted([80, 172, 264, 355], data.yearday)
data['season'] = seasons[f]

fig, ax = plt.subplots(figsize=(10,6))
data.groupby('season')['AEP_MW'].mean().plot.bar()
res = dict(tuple(data.groupby('season')))
res, res.keys()
fall_data = res['Fall']
spring_data = res['Spring']
summer_data = res['Summer']
winter_data = res['Winter']
fig, ax = plt.subplots(figsize=(10,6))

winter_data.groupby('year')['AEP_MW'].mean().plot.bar()
concat_data = [fall_data, spring_data, summer_data]
for_train = pd.concat(concat_data)
for_train = for_train.drop('season',1)
for_train = for_train.reset_index()
for_train = for_train.drop('index',1)
for_train
for_test = winter_data.drop('season', 1)
for_test
dataset1 = for_train.AEP_MW.values.reshape(-1,1)

sclar = MinMaxScaler(feature_range=(0,1))
dataset1 = sclar.fit_transform(dataset1)

dataset11 = for_test.AEP_MW.values.reshape(-1,1)

sclar = MinMaxScaler(feature_range=(0,1))
dataset11 = sclar.fit_transform(dataset11)

dataset1.shape, dataset11.shape
# convert an array of values into a dataset matrix
def create_dataset(dataset, seq_len):
    dataX, dataY = [], []
    for i in range(seq_len, len(dataset)):
        appnd = dataset[i - seq_len: i, 0]
        dataX.append(appnd)
        dataY.append(dataset[i,0])
    return np.array(dataX), np.array(dataY)
seq_len = 20 #choose sequence length

trainX, trainY = create_dataset(dataset1, seq_len)
testX, testY = create_dataset(dataset11, seq_len)
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1],1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

trainX.shape, testX.shape
model_s = tf.keras.Sequential()
model_s.add(tf.keras.layers.Bidirectional(LSTM(60, return_sequences=True, input_shape=(trainX.shape[1],1))))
model_s.add(Bidirectional(LSTM(20, return_sequences= False, activation= 'linear')))
model_s.add(tf.keras.layers.Dropout(0.5))
model_s.add(Dense(1))
# model.add(layers.LSTM(256))

model_s.compile(loss='MSE', optimizer = 'adam')
hist_s = model_s.fit(trainX, trainY, epochs=10, validation_split=0.3, batch_size=1000)
model_s.summary()
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14,6))
ax.plot(hist_s.history['loss'], 'b' ,label = 'train loss', linewidth=2)
ax.plot(hist_s.history['val_loss'], 'r', label ='Validation loss', linewidth=2)
ax.set_title('model loss')
ax.set_ylabel('mse')
ax.set_xlabel('epoch')
ax.legend()
plt.show()
# compute mean sequare error
predictions = model_s.predict(testX)

mse = mean_squared_error(testY, predictions)
print("mean square error = ", mse)
def plot_predictions(test, predicted, title):
    plt.figure(figsize=(16, 4))
    plt.plot(test, color='blue', label='Actual power consumption data')
    plt.plot(predicted, alpha=0.7, color='red', label='Predicted power consumption data')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.legend()
    plt.show()


plot_predictions(testY, predictions, "Predictions made by model")

