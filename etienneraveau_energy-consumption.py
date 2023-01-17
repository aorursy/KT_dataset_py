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
# Read data and look at features
data = pd.read_csv('/kaggle/input/hourly-energy-consumption/AEP_hourly.csv')

print(data.columns)
print(data.head)
# Preprocess data

from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta
import datetime

# Sort data by datetime
data = data.sort_values(by=['Datetime'])

# Convert dates columns into datetime64 type
data['Datetime'] = data['Datetime'].astype('datetime64')

# Conserve only last 3 years of data 
data = data.loc[data['Datetime']>=data['Datetime'][len(data['Datetime'])-1]-relativedelta(years=3)]
data.reset_index(inplace=True)
print(data)

# Scale and center data
scaler = MinMaxScaler()
consumption = scaler.fit_transform(np.reshape(data['AEP_MW'].values,(-1,1)))[:,0]
print(consumption)
# Create training and testing sets 
ratio = 0.8
split = (int)(np.floor(ratio*len(data)))
input_length = 20

x_train = [consumption[i-input_length:i] for i in range(input_length,split)]
x_test = [consumption[i-input_length:i] for i in range(input_length+split,len(consumption))]
y_train = consumption[input_length:split]
y_test = consumption[input_length+split:]

# Reshape x_train and x_test to fit lstm input
x_train_lstm = np.reshape(x_train,(np.shape(x_train)[0],np.shape(x_train)[1],1))
x_test_lstm = np.reshape(x_test,(np.shape(x_test)[0],np.shape(x_test)[1],1))
# Implement lstm forecaster

# Import deep learning libraries
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense

# Build our model
lstm = Sequential()
 
# Declare the layers
layers = [LSTM(units=128, input_shape=(input_length,1), activation='sigmoid',return_sequences=True),
          LSTM(units=128, activation='sigmoid'),
         Dense(1)]
 
# Add the layers to the model
for layer in layers:
    lstm.add(layer)

# Compile our model
lstm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
 
# Fit the model
history_lstm = lstm.fit(x_train_lstm, y_train, validation_data=(x_test_lstm,y_test), epochs=3, batch_size=32)
# Plot accuracy and loss evolutions over training 
import matplotlib.pyplot as plt

plt.figure()

plt.subplot(121)
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')  
plt.xlabel('epochs')
plt.legend(['train','val'], loc='upper left')

plt.subplot(122)
plt.plot(history_lstm.history['accuracy'])
plt.plot(history_lstm.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')  
plt.xlabel('epochs')
plt.legend(['train','val'], loc='upper left')

plt.show()
# Print predictions
predictions = lstm.predict(x_test_lstm)
first_date = data['Datetime'][len(data)-len(y_test)]
predicted_dates = [first_date + datetime.timedelta(hours=i) for i in range(len(x_test))]
plt.figure()
plt.plot(data['Datetime'],scaler.inverse_transform(np.reshape(consumption,(-1,1))),color='b',alpha=0.7)
plt.plot(predicted_dates,scaler.inverse_transform(np.reshape(predictions,(-1,1))),color='r',alpha=0.4)
plt.xlabel('Datetime')
plt.ylabel('Energy consumption in MegaWatt')
plt.title('American energy consumption evolution over time')
plt.legend(['true data','prediction'])
plt.show()