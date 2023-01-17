import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
#importing the data

train=pd.read_csv('../input/googledta/trainset.csv')
train.shape
train.head()
train.tail()
train_set=train.iloc[:,2:3].values

train_set
sc=MinMaxScaler(feature_range=(0,1))
train_scalar=sc.fit_transform(train_set)
X_train=[]

y_train=[]

for i in range(60,len(train_scalar)-1):

    X_train.append(train_scalar[i-60:i,0])

    y_train.append(train_scalar[i,0])

x_train,y_train=np.array(X_train),np.array(y_train)
x_train[1]
y_train[1]
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train[1]
from keras.models import Sequential

from keras.layers import Dropout,LSTM,Dense
model=Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))

model.add(Dropout(0.3))

model.add(LSTM(units=50))

model.add(Dropout(0.2))

model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100,batch_size=64)
test=pd.read_csv('../input/googledta/testset.csv')

test.head()
real=test.iloc[:,2:3].values
dataset_total=pd.concat((train['High'],test['High']),axis=0)

inputs = dataset_total[len(dataset_total) - len(test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []

for i in range(60,len(test)+59):

    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = model.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Visualising the results

plt.figure(figsize=(12,10))

plt.plot(real, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()