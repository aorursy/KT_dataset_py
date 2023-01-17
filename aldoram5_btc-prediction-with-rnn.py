# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data= pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-03-27.csv")
data.info()
data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date
data.info()
data.head()
data['hour'] = pd.to_datetime(data['Timestamp'],unit='s').dt.hour
daily = data.groupby('date')
daily = daily['Weighted_Price'].mean()
hourly = data.groupby(['date','hour'])
hourly = hourly['Weighted_Price'].mean()
prediction_days = 30
df_train= daily[:len(daily)-prediction_days]
df_test= daily[len(daily)-prediction_days:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))
# Importing the Keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LeakyReLU
regressor = Sequential()
regressor.add(LSTM(units = 4, activation = 'tanh', input_shape = (None, 1)))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
regressor.save("./btc-model.h5")
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(predicted)
sns.set_style("darkgrid")
plt.figure(figsize=(25,15), dpi=80)
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'BTC Price')
plt.plot(predicted, color = 'blue', label = 'Predicted BTC Price')
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['date']
plt.xticks(x, labels, rotation = 'vertical')
plt.xlabel('Time', fontsize=40)
plt.ylabel('USD', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()
prediction_hours = 30
df_train= hourly[:len(hourly)-prediction_hours]
df_test= hourly[len(hourly)-prediction_hours:]
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))
regressor = Sequential()
regressor.add(LSTM(units = 4, activation = 'tanh', input_shape = (None, 1)))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
regressor.save("./btc-model-hourly.h5")
test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted = regressor.predict(inputs)
predicted = sc.inverse_transform(predicted)
sns.set_style("darkgrid")
plt.figure(figsize=(25,15), dpi=80)
ax = plt.gca()  
plt.plot(test_set, color = 'red', label = 'BTC Price')
plt.plot(predicted, color = 'blue', label = 'Predicted BTC Price')
df_test = df_test.reset_index()
x=df_test.indexinih
labels = df_test['hour']
plt.xticks(x, labels, rotation = 'vertical')
plt.xlabel('hour', fontsize=40)
plt.ylabel('$USD', fontsize=40)
plt.legend(loc=2, prop={'size': 25})
plt.show()