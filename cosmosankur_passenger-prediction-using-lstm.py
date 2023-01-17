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

import matplotlib.pyplot as plt

import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM

from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('/kaggle/input/air-passengers/AirPassengers.csv')

dataset = dataset['#Passengers']

dataset = np.array(dataset).reshape(-1,1)
dataset
plt.plot(dataset)
scaler = MinMaxScaler()

dataset = scaler.fit_transform(dataset)
dataset[5]
train_size = 100

test_size = 44
train = dataset[0:train_size,:]

test = dataset[train_size:len(dataset),:]
dataset[-1]
def get_data(dataset,look_back):

    dataX , dataY =[] , []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back),0]

        dataX.append(a)

        dataY.append(dataset[i+look_back,0])

    return np.array(dataX),np.array(dataY)
look_back = 1
x_train ,y_train = get_data(train,look_back)
x_test,y_test = get_data(test,look_back)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
model = Sequential()

model.add(LSTM(5,input_shape = (1,look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()
model.fit(x_train,y_train,epochs=50,batch_size=1)
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)

y_test = np.array(y_test)

y_test = y_test.reshape(-1,1)

y_test = scaler.inverse_transform(y_test)
plt.plot(y_test, label = 'real number of passengers')

plt.plot(y_pred,label='predicted number of passengers')

plt.ylabel('passengers')

plt.legend()

plt.show()