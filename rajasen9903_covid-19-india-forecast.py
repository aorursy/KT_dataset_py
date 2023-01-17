# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

!pip install pandas==1.0.3
!pip install tensorflow==2.0

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

print(pd.__version__)
print(np.__version__)
print(tf.__version__)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#read the latest csv value
df =pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#select only India as country
filt_india = (df['Country/Region'] == 'India') 
#overwrite dataframe : only want to work with India Case
df = df[filt_india]
df
#removing duplicate rows if any
df.drop_duplicates(inplace=True)
# dropping State because its NaN and Country since its fixed value : India
df.drop(columns=['Province/State', 'Country/Region'], inplace=True)
#dropping Last Update column since it won't help in forecasting
df.drop(columns='Last Update', inplace=True)
#set SNo as index
df.set_index('SNo', inplace=True)
#convert Object type of ObservationDate to DateTime
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
#Sorting the sequence based on the observation date
df.sort_values('ObservationDate', inplace=True)
df.columns
#visualize observation date with confirmed cases
data = df['Confirmed']
data.index = df['ObservationDate']
data.plot()
data = data.values
data = data.astype(np.float32)
data
def create_sequence(input_data, steps):
    i = 0
    x = []
    y = []
    while (i+steps) < len(input_data):
        x.append(input_data[i:i+steps])
        y.append(input_data[i+steps])
        i = i + 1
    return x, y
# test the create_sequence
input_data = [10, 20, 30, 40, 50, 60]
x,y = create_sequence(input_data, 3)
print(x)
print(y)
n_steps = 3
x, y = create_sequence(data, n_steps)
x = np.asarray(x)
y = np.asarray(y)
row_index = x.shape[0] - 1
x, x_test = x[:row_index], x[row_index]
y, y_test = y[:row_index], y[row_index]
print(y)
print(y_test)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
# define model
model = Sequential()
model.add(LSTM(50, activation='relu',return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=1000, verbose=2)
from numpy import array
x_input = array([2.543e+03, 2.567e+03, 3.082e+03])
x_input = x_input.astype(np.float32)
x_input = x_input.reshape((1, n_steps, n_features))
x_input
x_test = x_test.reshape((1, n_steps, n_features))
yhat = model.predict(x_test, verbose=0)
print(yhat)
print(y_test)
