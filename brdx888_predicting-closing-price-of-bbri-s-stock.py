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

import pandas_datareader as web

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
# BBRI

bbri = web.DataReader('bbri.jk', 'yahoo', '2008-01-01', '2020-05-31')

bbri
# Let's visualize the Closing Price



rolling_mean20 = bbri['Close'].rolling(window=20).mean()

rolling_mean50 = bbri['Close'].rolling(window=50).mean()

rolling_mean200 = bbri['Close'].rolling(window=200).mean()



plt.figure(figsize = (16,8))

plt.plot(bbri['Close'], color = 'b', label = 'BBRI Close')

plt.plot(rolling_mean20, color = 'r', linewidth = 2.5, label = 'MA20')

plt.plot(rolling_mean50, color = 'y', linewidth = 2.5, label = 'MA50')

plt.plot(rolling_mean200, color = 'c',linewidth = 2.5, label = 'MA200')



plt.xlabel('Date', fontsize = 15)

plt.ylabel('Closing Price in Rupiah (Rp)', fontsize = 18)

plt.title('Closing Price of BBRI')

plt.legend(loc = 'lower right')
# Don't forget to using 'values' attribute before apply it into Neural Network!



X = bbri.drop('Close', axis = 1).values

y = bbri['Close'].values
from sklearn.model_selection import train_test_split
# Split the Data into Train and Test dataset with 20% test size.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train.shape
# Scale it with MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)
# Reshape it into 3-dimensional before input it into LSTM Model.

# Reshape the data to be 3-dimensional in the form [number of samples, number of time steps, and number of features].



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, LSTM
# just to remember our dataset shape



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
model = Sequential()



# input layer

model.add(LSTM(5, return_sequences=True, input_shape=(X_train.shape[1],1)))

model.add(LSTM(5, return_sequences= False))



# hidden layer

model.add(Dense(5, activation='relu'))

model.add(Dense(5, activation='relu'))



# output layer

model.add(Dense(1))



# compiler

model.compile(optimizer='adam', loss = 'mse')
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 25)
# Let's train our model!



model.fit(x = X_train,

          y = y_train,

          validation_data = (X_test, y_test),

          epochs = 500, 

          callbacks = [early_stop])
# So there are 162 iterations. Let's recap our training loss vs validation loss, and make it to DataFrame.



model_loss = pd.DataFrame(model.history.history)

model_loss
plt.figure(figsize = (12,6))

model_loss.plot()

plt.xlabel('n of Epochs')
from sklearn.metrics import mean_squared_error, explained_variance_score
predictions = model.predict(X_test)
# MSE

mean_squared_error(y_test, predictions)
# RMSE

np.sqrt(mean_squared_error(y_test, predictions))
# Explained variance regression score function



explained_variance_score(y_test, predictions)

# Best possible score is 1.0, lower values are worse (sklearn).
plt.figure(figsize=(12,6))

plt.scatter(y_test,predictions)

plt.plot(y_test, y_test, 'r', linewidth = 2.5)

plt.xlabel('y_test')

plt.ylabel('predictions')

plt.title('LSTM Model Prediction Evaluation')
# BBRI

bbri2 = web.DataReader('bbri.jk', 'yahoo', '2020-06-01', '2020-06-01')

bbri2
bbri2 = bbri2.drop('Close', axis = 1)
bbri2 = bbri2.values
bbri2 = scaler.transform(bbri2)
bbri2 = bbri2.reshape(-1, 5, 1)
model.predict(bbri2)