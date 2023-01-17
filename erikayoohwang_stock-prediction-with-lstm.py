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
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
df=pd.read_csv('/kaggle/input/nyse/prices-split-adjusted.csv')
df.info()
df.head()
# number of different stocks
print('number of different stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol)))
df.isnull().sum()
df.describe()
df_eqix=df[df.symbol=='EQIX']
df_eqix=df_eqix.set_index('date')
df_eqix.head()
# visualizing 'EQIX' stock price and volume as time
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)

plt.plot(df[df.symbol == 'EQIX'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'EQIX'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'EQIX'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'EQIX'].high.values, color='black', label='high')
plt.title('stock price')
plt.xlabel('time [days]')
plt.ylabel('price')
plt.legend(loc='best')

plt.subplot(1,2,2);
plt.plot(df[df.symbol == 'EQIX'].volume.values, color='black', label='volume')
plt.title('stock volume')
plt.xlabel('time [days]')
plt.ylabel('volume')
plt.legend(loc='best')
plt.show()
# standardization for data
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    return df

# function to create train, validation, test data given stock data and sequence length

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

def load_data(stock, seq_len):
    data_raw=stock.values #convert to numpy array
    data=[]
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw)-seq_len):
        data.append(data_raw[index: index+seq_len])
        
    data=np.array(data)
    valid_set_size=int(np.round(valid_set_size_percentage/100 * data.shape[0]));
    test_set_size=int(np.round(test_set_size_percentage/100 * data.shape[0]));
    train_set_size=data.shape[0]-(valid_set_size + test_set_size);
    
    x_train= data[:train_set_size, :-1, :]
    y_train= data[:train_set_size, -1, :]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]
# choose one stock
df_stock=df[df.symbol== 'EQIX'].copy()
df_stock.drop(['symbol'],1,inplace=True) # df_stock은 EQIX symbol만 사용
df_stock.drop(['volume'],1,inplace=True)
df_stock.drop(['date'],1,inplace=True)

cols=list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm= df_stock.copy()
df_stock_norm=normalize_data(df_stock_norm)

# create train, test data
seq_len=20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)
plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.open.values, color='red', label='open')
plt.plot(df_stock_norm.close.values, color='green', label='low')
plt.plot(df_stock_norm.low.values, color='blue', label='low')
plt.plot(df_stock_norm.high.values, color='black', label='high')
#plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
plt.title('stock')
plt.xlabel('time [days]')
plt.ylabel('normalized price/volume')
plt.legend(loc='best')
plt.show()
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
lstm_model=Sequential()

lstm_model.add(LSTM(input_shape=(19,4), units=50, return_sequences=True)) # adding LSTM layer
lstm_model.add(Dropout(0.2)) #adding Dropout

lstm_model.add(LSTM(100, return_sequences = False))                            #Adding LSTM layer
lstm_model.add(Dropout(0.2))                                                   #Adding Dropout

lstm_model.add(Dense(units=4))                                                 #Adding Dense layer with activation = "linear"
lstm_model.add(Activation('linear'))

'''Compiling the model'''

lstm_model.compile(loss='mse', optimizer='rmsprop')
'''Fitting the dataset into the model'''

lstm_model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_valid, y_valid))

'''Predicted values of train/val/test dataset'''

train_y_pred = lstm_model.predict(x_train)
print('prediction of train data is ',train_y_pred)
val_y_pred = lstm_model.predict(x_valid)
print('prediction of validation data is ', val_y_pred)
test_y_pred = lstm_model.predict(x_test)
print('prediction of test data is ', test_y_pred)
# visualizing the trained/predicted/test dataset
c = 0 # 0 = open, 1 = close, 2 = high, 3 = low

plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:, c], color='blue', label='Train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_valid.shape[0]), y_valid[:, c], 
         color='gray', label='Validation target')

plt.plot(np.arange(y_train.shape[0]+ y_valid.shape[0], y_train.shape[0]+y_test.shape[0]+y_test.shape[0]), 
         y_test[:, c], color='black', label='Test target')

plt.plot(np.arange(train_y_pred.shape[0]),train_y_pred[:, c], color='red', label='Train Prediction')

plt.plot(np.arange(train_y_pred.shape[0], train_y_pred.shape[0]+val_y_pred.shape[0]), 
         val_y_pred[:, c], color='orange', label='Validation Prediction')

plt.plot(np.arange(train_y_pred.shape[0]+val_y_pred.shape[0], 
                   train_y_pred.shape[0]+val_y_pred.shape[0]+test_y_pred.shape[0]), 
         test_y_pred[:, c], color='green', label='Test Prediction')


plt.title('Past and Future Stock Prices')
plt.xlabel('Time [days]')
plt.ylabel('Normalized Price')
plt.legend(loc='best');

plt.subplot(1,2,2);
plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]), y_test[:, c], color='black', label='Test target')

plt.plot(np.arange(train_y_pred.shape[0], train_y_pred.shape[0]+test_y_pred.shape[0]), test_y_pred[:, c], 
         color='green', label='Test Prediction')

plt.title('Future Stock Prices')
plt.xlabel('Time [days]')
plt.ylabel('Normalized Price')
plt.legend(loc='best');

train_acc = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]), 
                            np.sign(train_y_pred[:,1]-train_y_pred[:,0])).astype(int)) / y_train.shape[0]
val_acc = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]), 
                          np.sign(val_y_pred[:,1]-val_y_pred[:,0])).astype(int)) / y_valid.shape[0]
test_acc = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]), 
                           np.sign(test_y_pred[:,1]-test_y_pred[:,0])).astype(int)) / y_test.shape[0]

print('Accuracy for Close - Open price for Train/Validation/Test Set: %.2f/%.2f/%.2f'%(train_acc, val_acc, test_acc))
