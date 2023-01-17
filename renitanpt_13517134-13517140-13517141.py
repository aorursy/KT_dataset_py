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
stock = pd.read_csv("/kaggle/input/if4074-praktikum-2-rnn/train_IBM.csv")
stock.head
stock.isnull().sum()
# drop nan value
stock.dropna(inplace=True)
# Choose close column as feature
harga = stock.iloc[:,4:5].values
harga
# scaling data to 0 - 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 
harga_scaled = scaler.fit_transform(harga)
# create feature training data
seq_len = 321
X_train = []
y_train = []

for i in range(seq_len, len(stock)):
    X_train.append(harga_scaled[i-seq_len : i, 0])
    y_train.append(harga_scaled[i, 0])

X = np.array(X_train)
y = np.array(y_train)
# split training and validation data
train_size = 2139
X_val = X[train_size:]
X_train = X[:train_size]
y_val = y[train_size:]
y_train = y[:train_size]
#reshape input data for RNN model
X_train = np.reshape(X_train, (train_size, seq_len, 1))
X_val = np.reshape(X_val, (len(X_val), seq_len, 1))
print(X_train.shape)
print(X_val.shape)
from keras.layers import SimpleRNN, Dense, LSTM, Dropout
from keras.models import Sequential

model = Sequential()

model.add(SimpleRNN(50, return_sequences=True))
model.add(Dropout(0.15))

model.add(SimpleRNN(25, return_sequences=True))
model.add(Dropout(0.15))

model.add(Dense(1, activation="linear"))
from keras import backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# model.compile(optimizer="adam",loss="MSE")
model.compile(optimizer='rmsprop', loss='mse', metrics =[rmse])
# fit training data
model.fit(X_train,y_train,epochs=20,batch_size=25)
# predict val data
rnn_predictions = model.predict(X_val)

result = []
for i in range(len(rnn_predictions)):
    result.append(rnn_predictions[i][320][0])
import math
from sklearn.metrics import mean_squared_error

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    
    return rmse
metric = return_rmse(y_val,result)
metric
# Predict test data
X_test = np.reshape(rnn_predictions[237], (1, seq_len, 1))
rnn_predictions = model.predict(X_test)
rnn_predictions = scaler.inverse_transform(rnn_predictions[0])
# Write to csv

pred = pd.read_csv("/kaggle/input/if4074-praktikum-2-rnn/sample_submission.csv")
pred['Close'] = rnn_predictions
pred.to_csv ('submission.csv', index=False, header=True)