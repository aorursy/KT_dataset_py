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
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import LSTM 

import pandas
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math 
data=pandas.read_csv(r"../input/airlinepassengers1csv/airline-passengers1.csv")
data.head(10)
data=data.iloc[:,1]
data.tail()
dataset=data.astype('float32')
dataset=dataset.values
type(dataset)
data.shape
dataset=dataset.reshape(144,1)
dataset.shape
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)
dataset.shape
dataset
train_size=int(len(dataset)*0.7)
test_size=int(len(dataset)-train_size)
train=dataset[test_size:len(dataset),:]
test=dataset[0:test_size,:]
print(len(train),len(test))
print(train.shape)
print(test.shape)
train
import numpy
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
look_back=1
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
trainX=numpy.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX=numpy.reshape(testX,(testX.shape[0],1,testX.shape[1]))
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
model=Sequential()
model.add(LSTM(4,input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,epochs=1000)
trainPredict = model.predict(trainX)
print(trainPredict)
testPredict = model.predict(testX)
print([trainY])
trainPredict = scaler.inverse_transform(trainPredict)
print(trainY.shape)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
trainPredict=scaler.inverse_transform(trainPredict)
plt.plot(trainPredict)
plt.show()


testPredict=scaler.inverse_transform(testPredict)
plt.plot(testPredict)
plt.show()
plt.plot(trainPredict,color='orange')
plt.plot(testPredict,color='black')
plt.show()

trainPredict.shape
testPredict
testY
