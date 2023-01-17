# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
import tensorflow.keras.backend as K
import tensorflow as tf

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
with open('../input/bse30-daily-market-price-20082018/BSE_30.csv', 'r') as f:
    
    header = f.readline()
    header = header.strip().split(',')
    
    symbol_index = header.index('Symbol')
    date_index = header.index('Date')
    open_index = header.index('Open')
    high_index = header.index('High')
    low_index = header.index('Low')
    close_index = header.index('Close')
    
    unique_symbols = set()
    total_records = 0
    
    return_pos = f.tell()
    line = f.readline()
    record = line.strip().split(',')
    
    min_date = datetime.datetime.strptime(record[date_index], '%m/%d/%Y')
    max_date = datetime.datetime.strptime(record[date_index], '%m/%d/%Y')
    
    f.seek(return_pos)
    
    for line in f.readlines():
        record = line.strip().split(',')
        
        total_records += 1
        
        unique_symbols.add(record[symbol_index])
        
        date = datetime.datetime.strptime(record[date_index], '%m/%d/%Y')
        
        min_date = min(min_date, date)
        max_date = max(max_date, date)
    
    delta = max_date - min_date
    
    n_timesteps = delta.days + 1
    
    unique_symbols = list(unique_symbols)
    
    n_symbols = len(unique_symbols)
    
    n_features = 4
    
    data = np.zeros((n_timesteps, n_symbols, n_features))
    data.fill(-999)
    
    f.seek(0)
    
    f.readline()
    
    for line in f.readlines():
        record = line.strip().split(',')
        
        symbol = record[symbol_index]
        symbol_id = unique_symbols.index(symbol)
        
        date = datetime.datetime.strptime(record[date_index], '%m/%d/%Y')
        delta = date - min_date
        t = delta.days
        
        if not record[open_index] == 'null':
            data[t][symbol_id][0] = record[open_index]
        if not record[high_index] == 'null':
            data[t][symbol_id][1] = record[high_index]
        if not record[low_index] == 'null':
            data[t][symbol_id][2] = record[low_index]
        if not record[close_index] == 'null':
            data[t][symbol_id][3] = record[close_index]
X_val = []
y_val = []
for t in range(data.shape[0] - 5):
    X_val.append(data[t])
    
    y_curr = []
    
    for i in range(t + 1, t + 6):
        y_curr.append(data[i][0][0])
    
    y_val.append(y_curr)
X = X_val[:-1000]
y = y_val[:-1000]
X_val = np.array(X_val)
y_val = np.array(y_val)
X = np.array(X)
y = np.array(y)
X.shape
X = X.reshape(1, X.shape[0], X.shape[1] * X.shape[2])
y = y.reshape(1, y.shape[0], y.shape[1])
X_val = X_val.reshape(1, X_val.shape[0], X_val.shape[1] * X_val.shape[2])
y_val = y_val.reshape(1, y_val.shape[0], y_val.shape[1])
y.shape
def custom_mse(y_true, y_pred):
    mask = K.not_equal(y_true, -999)
    
    uncorrected_loss = K.square(y_pred - y_true)
    
    corrected_loss = uncorrected_loss * tf.cast(mask, 'float32')
    
    loss = K.sqrt(K.mean(corrected_loss))
    
    return loss
model = Sequential()
model.add(LSTM(1000, input_shape=(None, n_symbols * n_features), return_sequences=True))
model.add(TimeDistributed(Dense(5)))
model.compile(loss=custom_mse, optimizer='adam')
model.fit(X, y, validation_data=(X_val, y_val), epochs=100, batch_size=1, verbose=2)
model.fit(X, y, validation_data=(X_val, y_val), epochs=200, batch_size=1, verbose=2)
model.save_weights('/kaggle/working/lstm_model.h5')
import pickle

with open('/kaggle/working/X.pkl', 'wb') as f:
    pickle.dump(X, f)

with open('/kaggle/working/y.pkl', 'wb') as f:
    pickle.dump(y, f)
    
with open('/kaggle/working/X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)

with open('/kaggle/working/y_val.pkl', 'wb') as f:
    pickle.dump(y_val, f)
with open('/kaggle/working/total_stock_data_timesteps_symbols_features.pkl', 'wb') as f:
    pickle.dump(data, f)

df['Symbol'].nunique()
