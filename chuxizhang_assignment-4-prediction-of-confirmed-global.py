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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
dataset = pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
Hubei = dataset[dataset['Province/State'].isin(['Hubei'])]
print(Hubei)
import seaborn as sns; sns.set()
copy = dataset.copy()
corr = copy.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
from matplotlib import pyplot
confirmed_df = copy
countries=['China', 'Italy', 'Brazil', 'Canada', 'Germany']
t = confirmed_df.loc[confirmed_df['Country/Region']=='China'].iloc[0,4:]
s = pd.DataFrame({'China':t})
for c in countries:    
    s[c] = confirmed_df.loc[confirmed_df['Country/Region']==c].iloc[0,4:]
pyplot.plot(range(t.shape[0]), s)
data = Hubei.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'], axis = 1)
data
value = data.columns.values
value
feature = data.values
feature
date = data.columns.values.tolist()
data_new = {'population': feature[0]}
df = pd.DataFrame(data_new)
df.index = date
df
df.values
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
data = scaler.fit_transform(df.values)
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
from sklearn.model_selection import train_test_split as tts

X, y = create_dataset(data, 10)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
from keras.layers import Dropout
model = Sequential()
model.add(LSTM(128, input_shape=(10, 1), return_sequences = True, activation = 'relu'))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(128))
model.add(Dense(1))
import tensorflow as tf
learning_rate = tf.keras.callbacks.ReduceLROnPlateau('value_loss', patience = 3, factor = 0.3, min_lr = 0.00001)

model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae'])
model.fit(X_train, y_train, epochs = 200, batch_size = 20, validation_data = (X_test, y_test), validation_freq = 10, callbacks = [learning_rate])
predictions = [df.values[-1]]
new = data.copy()

for i in range(20):
    prediction = model.predict(new[-10:,:].reshape(1, 10, 1))
    predictions.append(float(prediction * predictions[-1]))
    new = np.append(new, prediction, axis = 0)
data_predict = list(range(88, 88 + 20))
plt.plot(df.values)
plt.plot(data_predict, predictions[1:])
plt.legend(['Origin', 'Prediction'], loc = 'lower right')

Ontario = dataset[dataset['Province/State'].isin(['Ontario'])]
data_o = Ontario.drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'], axis = 1)
data_o
feature = data_o.values
date_r = data_o.columns.values.tolist()
data_n = {'population': feature[0]}
df = pd.DataFrame(data_n)
df.index = date
df
sca = MinMaxScaler(feature_range = (0, 1))
data = sca.fit_transform(df.values)
X, y = create_dataset(data, 10)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential()
model.add(LSTM(128, input_shape=(10, 1), return_sequences = True, activation = 'relu'))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(128))
model.add(Dense(1))
import tensorflow as tf
learning_rate = tf.keras.callbacks.ReduceLROnPlateau('value_loss', patience = 3, factor = 0.3, min_lr = 0.00001)

model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mae'])
model.fit(X_train, y_train, epochs = 200, batch_size = 20, validation_data = (X_test, y_test), validation_freq = 10, callbacks = [learning_rate])
predictions = [df.values[-1]]
new = data.copy()

for i in range(15):
    prediction = model.predict(new[-10:,:].reshape(1, 10, 1))
    predictions.append(float(prediction * predictions[-1]))
    new = np.append(new, prediction, axis = 0)
    
data_predict = list(range(88, 88 + 15))
plt.plot(df.values)
plt.plot(data_predict, predictions[1:])
plt.legend(['Origin', 'Prediction'], loc = 'upper left')