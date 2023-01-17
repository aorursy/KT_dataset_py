# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/AirPassengers.csv")
df.tail()
df = df['#Passengers']
df.head()
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(df)
plt.show()
def parse(dataset, pick = 1):
    X = list()
    Y = list()
    for i in range(len(dataset) - pick):
        X.append(dataset[i:i+pick][0])
        Y.append(dataset[i+pick])
    return np.array(X), np.array(Y) 
x, y = parse([1,2,3,4,5], pick = 1)
print(x)
print(y)
A = [1,2,3,4,5]
print(A[:2])
print(A[2])
print(A[2:])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df.values.astype('float32').reshape(-1,1))

def train_test(dataset, test_size = 0.2):
    ratio = int((1 - test_size) * len(dataset))
    return dataset[:ratio, :], dataset[ratio:,:]
    
train, test = train_test(df, test_size = 0.2)
Xtrain, ytrain = parse(train)
Xtest, ytest = parse(test)

pXtest = pd.Series([p[0] for p in Xtest])
pXtest.index = list(range(len(df) - len(test) + 1, len(df)))
plt.plot(Xtrain.reshape(-1,1))
plt.plot(pXtest)
plt.show()
from keras.layers import LSTM, Dense
from keras.models import Sequential
model = Sequential()
model.add(LSTM(256, input_shape = (1,1)))
model.add(Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = 'adam',
             metrics = ['accuracy'])
model.summary()
Xtrain.shape
Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, Xtrain.shape[1])
Xtest = Xtest.reshape(Xtest.shape[0], 1, Xtrain.shape[1])
Xtrain.shape
model.fit(Xtrain, ytrain, epochs = 100, batch_size = 16)
y_pred = model.predict(Xtest)
pytest = scaler.inverse_transform(model.predict(Xtest)) 
pytest = pd.Series([p[0] for p in pytest])
pytest.index = list(range(len(df) - len(test) + 1, len(df)))
plt.plot(scaler.inverse_transform(df))
plt.plot(scaler.inverse_transform(model.predict(Xtrain)))
plt.plot(pytest)
plt.show()
y_pred.shape
