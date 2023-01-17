# importing necessary libraries 
import numpy as np
import pandas as pd
import math

from sklearn.metrics import mean_squared_error
from shapely.geometry import Point
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta

from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing Data
Data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
Data.head()
Data.shape
Data.Date.value_counts()
# check null values in dataset
Data.isnull().sum()
# Checking number of data records under each country
Data.Country_Region.value_counts()
countries = Data['Country_Region'].unique()
print(f'{len(countries)} countries are in dataset:\n{countries}')
group_date_max = Data.Date.value_counts().max()
group_date_min = Data.Date.value_counts().min()
print(group_date_max,group_date_min)
min_date = Data.Date.min()
max_date = Data.Date.max()
print(min_date,max_date)
Data.describe()
confirmed_total_date_noChina = Data[Data['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})
fatalities_total_date_noChina = Data[Data['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})
total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)

fig, (ax1) = plt.subplots(1, figsize=(10,5))
total_date_noChina.plot(ax=ax1)
ax1.set_title("Global confirmed cases excluding China", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)

total_date_noChina.ConfirmedCases.min()
total_date_noChina
total_date_noChina = total_date_noChina.sort_values('Date',ascending=True)

total_date_noChina
# creating input data 
input_data = total_date_noChina.iloc[:,0:1].values
# getting total record count to create train and test data test
records = total_date_noChina.count()
# train data set which is 3 less than total data set
records = records[0] - 7
train = total_date_noChina.iloc[0:records,0:1].values

# test data set with 10 records
test = total_date_noChina.iloc[records:,0:1].values
#print(input_data.shape)
print(input_data.shape,train.shape,test.shape)
# multi-step data preparation
from numpy import array
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y1 = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
    # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y1.append(seq_y)
    return array(X), array(y1)

# define input sequence
raw_seq = train
# choose a number of time steps
n_steps_in, n_steps_out = 3, 7
# split into samples
X, y1 = split_sequence(raw_seq, n_steps_in, n_steps_out)
# summarize the data
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y1 = y1.reshape(y1.shape[0], y1.shape[1])
# Model Building

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=((X.shape[1],1))))
model.add(LSTM(50, activation='relu', return_sequences=True,))
model.add(LSTM(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y1, epochs=1000, batch_size = 30, verbose = 0)
# Model testing to forcast 7 consecutive day wich will be compared with test data actuals
x_input = train[(records - n_steps_in):records,0:1]
x_input = x_input.reshape((1, n_steps_in, n_features))
test_predicted = model.predict(x_input, verbose=0)
test_predicted = test_predicted.reshape(n_steps_out,)
test_predicted1 = pd.Series(test_predicted)
test = test.reshape(n_steps_out,)
test1 = pd.Series(test)
pd.concat([test1,test_predicted1], axis=1)

plt.plot(test, color= 'red', label = 'test_data')
plt.plot(test_predicted, color= 'blue', label = 'predicted_test_data')
plt.title('Test Data Forecast')
plt.xlabel('time')
plt.ylabel('Confirmed_Cases')
plt.legend()
# Forcast Confirmed Cases for 7 consecutive days
#x_input = input_data[-n_steps_in:]
x_input = test[-3:]
x_input = x_input.reshape((1, n_steps_in, n_features))

forecast = model.predict(x_input, verbose=0)

#forecast = test_predicted.reshape(n_steps_in,1)
forecast
# Plotting Forecast Data
maximum_date = Data.Date.max()

date = pd.date_range(maximum_date, periods=8, closed='right')
date
date = pd.Series(date)
forecast1 = forecast.reshape(n_steps_out,)
forecast2 = pd.Series(forecast1)
forcast_data = pd.concat([date,forecast2], axis=1)
forcast_data.columns = ['Date','Forecast_Corfirmed_Cases']
plt.figure(figsize=(10,5))
plt.plot(date,forecast2)
plt.title('7 Days Forecast')
plt.xlabel('Time')
plt.ylabel('Confirmed_Cases')
print(forcast_data)