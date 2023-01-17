import numpy as np # linear algebra
import pandas as pd 
import tensorflow as tf# data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt

#get the dataset

confirm_temp = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

cols = confirm_temp.keys()

confirmed = confirm_temp.loc[:, cols[4]:cols[-1]]


confirmed.head()
#plot for the true confirmed people
dates = confirmed.keys()
total_confirmed = []

for i in dates:
    total_confirmed.append(confirmed[i].sum())
    
fig = plt.figure(figsize=(12,4))
plt.xlabel('dates')
plt.ylabel('number')
plt.plot(range(0,len(dates)), total_confirmed, color='black', label='confirm') 
plt.legend(loc = 'upper right')
plt.show()
#use MinMaxScaler to standardlize data into (0,1)
days = np.array([i for i in range(len(dates))])
total_confirmed = np.array(total_confirmed)
temp_total = total_confirmed.copy()
scaler = MinMaxScaler(feature_range=(0, 1))
total_confirmed = scaler.fit_transform(total_confirmed.reshape(-1,1))
#seperate the data for predict, the timestep is 20 
X_train = []
y_train = []
for i in range(len(dates)-20-1):
    X_train.append(total_confirmed[i:i+20,0])
    y_train.append(total_confirmed[i+20:i+21,0])
X_train, y_train = np.array(X_train), np.array(y_train)
# reshape the data to meet the keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#use LSTM model to train the data
regressor = Sequential()

regressor.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 128, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 512))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 80, batch_size = 32)
predict_list = []
predict_step = 20
#get the last 20th days to predict for a new day confirm number
for i in range(20):
    predict_list.append(total_confirmed[total_confirmed.shape[0]-(20-i):total_confirmed.shape[0]-(20-i)+1,0][0])
final_predict = []

#predict the confirm number trendency for the nest 20 days
for i in range(predict_step):
    temp_predict = np.array(predict_list)
    temp_predict = np.reshape(temp_predict,(1,20,1))
    temp = regressor.predict(temp_predict)
    temp = scaler.inverse_transform(temp)
    final_predict.append(temp[0][0])
    predict_list.append(temp[0][0])
    del predict_list[0]
# plot the true confirmed number and prediction
fig = plt.figure(figsize=(12,4))
plt.xlabel('dates')
plt.ylabel('number')
plt.plot(temp_total, color='black', label='actual') 
plt.plot(range(79,79+20),np.array(final_predict), color='red', label='predict')
plt.legend(loc = 'upper right')
plt.show()
    




