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
df = pd.read_csv("/kaggle/input/final-dataframe-may4/final_dataframe.csv", thousands=',').drop(columns=['Unnamed: 0'])
df = df[df['state_New Jersey'] ==1]
## New Jersey infected time series
df['cases']
features_considered = ['cases', 'deaths']
features = df[features_considered]
features.index = df['date_num']
features.head()
input_data = features.iloc[:,0:1].values  ### We only predict cases data
## Scale the input data, which is the infected time series
scaler = MinMaxScaler(feature_range=(0, 1))

scaler = scaler.fit(input_data)

input_data = scaler.transform(input_data)
### Record is the number of days in the series - 7, which is the break point for train test split
records = features.count()[0]-7
train = scaler.transform(features.iloc[0:records, 0:1].values)

test = scaler.transform(features.iloc[records:,0:1].values)
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
n_steps_in, n_steps_out = 4, 7
# split into samples
X, y1 = split_sequence(raw_seq, n_steps_in, n_steps_out)
# The parameter n_step_in is the size of the past window of information. 
# The n_steps_out is how far in the future does the model need to learn to predict. the label that needs to be predicted.

# Result：
# y1 is Tn+3
# X is Tn, Tn+1, Tn+2

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
model.fit(X, y1, epochs=1000, batch_size = 30, verbose = 1)
# Model testing to forcast 7 consecutive day wich will be compared with test data actuals
x_input = train[(records - n_steps_in):records,0:1]
x_input = x_input.reshape((1, n_steps_in, n_features))
test_predicted = model.predict(x_input, verbose=0)
test_predicted = test_predicted.reshape(n_steps_out,)
test_predicted1 = pd.Series(test_predicted)
test = test.reshape(n_steps_out,)
test1 = pd.Series(test)
compare = pd.concat([test1,test_predicted1], axis=1)
pd.DataFrame(scaler.inverse_transform(compare))
test = scaler.inverse_transform(test.reshape(7,1))
test_predicted = scaler.inverse_transform(test_predicted.reshape(7,1))
plt.plot(test, color= 'red', label = 'test_data')
plt.plot(test_predicted, color= 'blue', label = 'predicted_test_data')
plt.title('Test Data Forecast')
plt.xlabel('time')
plt.ylabel('Confirmed_Cases')
plt.legend()
X.shape
# Forcast Confirmed Cases for 7 consecutive days
#x_input = input_data[-n_steps_in:]
x_input = test[-4:]
x_input = x_input.reshape((1, 4, n_features))

forecast = model.predict(x_input, verbose=0)

#forecast = test_predicted.reshape(n_steps_in,1)
forecast
features
date = pd.date_range('2020-05-04', periods=8, closed='right')
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