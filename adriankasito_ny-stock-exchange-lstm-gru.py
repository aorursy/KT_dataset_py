# Load some required libraries
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
import keras
import matplotlib.pyplot as plt
import math
import time
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, r2_score
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.offline as pyo
cf.go_offline()
pyo.init_notebook_mode()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
print(__version__)
import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Load data
data = pd.read_csv('/kaggle/input/nyse/prices-split-adjusted.csv', parse_dates=['date'], index_col='date')
data.head(3)
#data['Month'] = data.index.month
#months = {1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November', 12:'December'}
#data['Month'] = data['Month'].map(months)
#data
#Check for duplicated values
data.duplicated().sum()
data.drop_duplicates(inplace=True)
#Check for null values
data.isna().sum()
data.symbol.value_counts()
plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
plt.plot(data[data.symbol=='MUR'].open.values, label='open', color='cyan', linewidth=0.8)
plt.plot(data[data.symbol=='MUR'].close.values, label='close', color='blue', linewidth=0.8)
plt.plot(data[data.symbol=='MUR'].high.values, label='high', color='darkred', linewidth=0.8)
plt.plot(data[data.symbol=='MUR'].low.values, label='low', color='brown', linewidth=0.8)
plt.legend(loc='best')
plt.xlabel('time[days]')
plt.ylabel('Prices')
plt.title('Stock price')


adrian = pd.DataFrame(data[data.symbol=='MUR'].volume.values)
adrian.iplot(kind='line', title='Volumes', yTitle='Prices', xTitle='time[days]', color='black')
ad = pd.DataFrame(data[data.symbol=='MUR'].close)
ad
training_set = ad[:'2016'].values
test_set = ad['2016':].values
training_set.shape
test_set.shape
# Scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(training_set.reshape(-1,1))
scaled_test = scaler.transform(test_set.reshape(-1,1))
scaled_data.shape
scaled_data
X_train = []
y_train = []
for i in range(60,1762):
    X_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshape for effecient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_train.shape
#LSTM architecture
model = Sequential()
# First LSTM layer with Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
# Second LSTM layer
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
# Third LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Fourth LSTM layer
model.add(LSTM(units=50))
model.add(Dropout(0.5))
# The output layer
model.add(Dense(units=50, kernel_initializer='uniform', activation='tanh'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

# Compiling the RNN
model.compile(optimizer='adam',loss='mean_squared_error')
# Fitting to the training set
start = time.time()
model.fit(X_train,y_train,epochs=200,batch_size=35, validation_split=0.05, verbose=1)
print ('compilation time : ', time.time() - start)
model.summary()
dataset_total = pd.concat((ad["close"][:'2016'],ad["close"]['2016':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set)-60 :].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []
y_test = []
for i in range(60,312):
    X_test.append(inputs[i-60:i,0])
    y_test.append(inputs[i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
# Inverse transform is to denormalize the predicted_stock_price
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
predicted_stock_price, test_set

predicted_stock_price.shape
plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
plt.plot(predicted_stock_price, linewidth=1.2, color='green', label='Predicted [MUR] Stock price')
plt.plot(test_set, linewidth=1.2, color='darkred', label='Real [MUR] Stock price')
plt.xlabel('Time', fontsize=8)
plt.ylabel('IBM Stock Price', fontsize=8)
plt.legend(loc='best', fontsize=10)
plt.show()

predicted_stock_price.shape
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
rmse
# R2_score for the LSTM
r2_score(test_set, predicted_stock_price)
GRU_model = Sequential()
# First GRU layer with Dropout regularisation
GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Second GRU layer
GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Third GRU layer
GRU_model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
GRU_model.add(Dropout(0.2))
# Fourth GRU layer
GRU_model.add(GRU(units=50, activation='tanh'))
GRU_model.add(Dropout(0.5))
# The output layer
#GRU_model.add(Dense(units=50, kernel_initializer='uniform', activation='tanh'))
GRU_model.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
# Compiling the RNN
GRU_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True),loss='mean_squared_error')
# Fitting to the training set
start = time.time()
GRU_model.fit(X_train,y_train,epochs=200,batch_size=35, validation_split=0.05, verbose=1)
print ('compilation time : ', time.time() - start)
GRU_model.summary()
X_test = []
for i in range(60,312):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
GRU_predicted_stock_price = GRU_model.predict(X_test)
# Denormalizing the predicted_stock_price
GRU_predicted_stock_price = scaler.inverse_transform(GRU_predicted_stock_price)
GRU_predicted_stock_price.shape
plt.figure(figsize=(8,6))
plt.subplot(1,1,1)
plt.plot(GRU_predicted_stock_price, linewidth=1.2, color='green', label='Predicted [MUR] Stock price')
plt.plot(test_set, linewidth=1.2, color='black', label='Real [MUR] Stock price')
plt.xlabel('Time', fontsize=8)
plt.ylabel(' stock Price', fontsize=8)
plt.legend(loc='best', fontsize=10)
plt.show()
Rmse = math.sqrt(mean_squared_error(test_set, GRU_predicted_stock_price))
Rmse
# R2_score for the GRU
r2_score(test_set, GRU_predicted_stock_price)
