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
df = pd.read_csv('/kaggle/input/nyse/prices.csv')
df.head(3)
df.shape
df = df.drop(columns=['date','symbol'])
df.head()
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import mean_squared_error, accuracy_score

plt.figure(figsize=(15,15))

plt.plot(df['close'])
scaler = MinMaxScaler(feature_range=(0,1))

df = scaler.fit_transform(df)   #Jadi numpy array, tidak lagi pandas



y_close = df[:,1]



plt.figure(figsize=(15,15))

plt.plot(y_close)

plt.show()
ntrain = int(len(y_close)*0.8) 



train = df[0:ntrain]

test  = df[ntrain:len(df)]



y_close_train = y_close[0:ntrain]

y_close_test  = y_close[ntrain:len(y_close)]
y_close_test.shape
import numpy as np



def to_sequences(seq_size, data,close):

    x = []

    y = []



    for i in range(len(data)-seq_size-1):

        window = data[i:(i+seq_size)]

        after_window = close[i+seq_size]

        window = [[x] for x in window]

        x.append(window)

        y.append(after_window)

        

    return np.array(x),np.array(y)





timesteps = 10



x_train, y_train = to_sequences(timesteps, train, y_close_train)

x_test, y_test   = to_sequences(timesteps, test, y_close_test)



print("Shape of x_train: {}".format(x_train.shape))

print("Shape of x_test: {}".format(x_test.shape))

print("Shape of y_train: {}".format(y_train.shape))

print("Shape of y_test: {}".format(y_test.shape))
x_train[0]
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[2], x_train.shape[1],x_train.shape[3]))

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[2],x_test.shape[1],x_test.shape[3]))
print(x_train.shape)
x_train[0][0].shape
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)

ax1.imshow(x_train[0][0])

ax2.imshow(x_train[1][0])

ax3.imshow(x_train[2][0])

ax4.imshow(x_train[3][0])

ax5.imshow(x_train[4][0])
import numpy as np

import pandas as pd



import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten, Dropout

from keras.optimizers import Adam

from keras.layers import Conv1D, Conv2D, MaxPooling2D

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import MinMaxScaler

import csv

import collections

from scipy.stats import zscore

from datetime import datetime

import matplotlib.pyplot as plt

cnn = Sequential()

cnn.add(Conv2D(8, kernel_size = (1, 2), strides = (1, 1),  padding = 'valid', 

               activation = 'relu', input_shape = (1,10,5)))

cnn.add(MaxPooling2D(pool_size = (1,2)))



cnn.add(Flatten())

cnn.add(Dense(64, activation="relu"))

cnn.add(Dropout(0.5))

cnn.add(Dense(1, activation="relu"))

cnn.summary()   



cnn.compile(loss='mean_squared_error', optimizer='nadam')



monitor = EarlyStopping(monitor='val_loss', min_delta=1, patience=2, verbose=2, mode='auto') 

checkpointer = ModelCheckpoint(filepath="CNN_Parameters.hdf5", verbose=0, save_best_only=True) # save best model
history = cnn.fit(x_train,y_train,validation_split=0.2,batch_size = 128, callbacks=[checkpointer],verbose=1,epochs = 100)
plt.plot(history.history['loss'], label = 'loss')

plt.plot(history.history['val_loss'], label = 'val loss')

plt.legend()
cnn.load_weights('CNN_Parameters.hdf5')



pred = cnn.predict(x_test)

print(pred.shape)
score = np.sqrt(metrics.mean_squared_error(y_test, pred))

print("RMSE Score: {}".format(score))
plt.figure(figsize=(15,15))



plt.plot(y_test, label = 'actual')

plt.plot(pred,   label = 'predicted')

plt.legend()

plt.show()