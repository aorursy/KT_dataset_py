# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
   print(dirname)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
stock_data_path = "/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/"
all_file = os.listdir(stock_data_path)
print('googl.us.txt' in all_file)
google_stock_df = pd.read_csv(os.path.join(stock_data_path,'googl.us.txt'),date_parser=True)
google_stock_df.head()
google_stock_df.tail()
google_stock_df.isnull().sum()
google_stock_df.OpenInt.unique()
google_stock_df.drop(['OpenInt'],axis=1,inplace=True)
google_stock_df.head()
google_stock_df.tail()
train_df = google_stock_df[google_stock_df['Date']<'2014-08-25']
test_df = google_stock_df[google_stock_df['Date']>='2014-08-25']
train_df=train_df.drop(['Date'],axis=1).copy()
test_df = test_df.drop(['Date'],axis=1).copy()
from sklearn.preprocessing import MinMaxScaler
train_scaler = MinMaxScaler()
test_scaler = MinMaxScaler()
train_array=train_scaler.fit_transform(train_df)
test_array=test_scaler.fit_transform(test_df)
train_array
X_train = []
Y_train = []
X_test = []
Y_test = []

for i in range(60,train_array.shape[0]):
    X_train.append(train_array[i-60:i])
    Y_train.append(train_array[i][0])
    
for i in range(60,test_array.shape[0]):
    X_test.append(test_array[i-60:i])
    Y_test.append(test_array[i][0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(14,8))
plt.plot(Y_train,color='blue')
plt.plot(Y_test,color='red')
plt.show()
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras import Sequential
regressor = Sequential()
regressor.add(LSTM(50,return_sequences = True,input_shape=X_train.shape[1:]))
regressor.add(Dropout(0.2))

regressor.add(LSTM(60,return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(80,return_sequences= True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(120))
regressor.add(Dropout(0.5))

regressor.add(Dense(1))
regressor.summary()
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=128,epochs=10)
result = regressor.predict(X_test)
result[:10]
test_scaler.scale_
scale = 1/1.81636545e-03
scale
result = result*scale
result[:10]
y_test = Y_test*scale
y_test[:10]
plt.figure(figsize=(14,8))
plt.plot(result,color = 'red',label='predicted')
plt.plot(y_test,color =  'blue', label = 'actual')
plt.legend()
plt.show