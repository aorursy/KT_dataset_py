# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pylab import plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
os.listdir('../input/')
data=pd.read_csv('../input/NSW_Demand.csv')
data.head()

from sklearn.preprocessing import MinMaxScaler
cols=data.columns
Demand=np.asarray(data[cols[1]])
Demand=np.reshape(Demand,(Demand.shape[0],1))
Price=np.asarray(data[cols[2]])
Price=np.reshape(Price,(Price.shape[0],1))

sc=MinMaxScaler(feature_range=(0,1))

Demand_scaled=sc.fit_transform(Demand)
Price_scaled=sc.fit_transform(Price)
plt.plot(Demand_scaled)
plt.title("Demand Data")
plt.figure()
plt.title("Price Data")
plt.plot(Price_scaled)

training_data=np.asarray([Demand_scaled,Price_scaled])
# print(training_data.shape)
# for 
X=[]
Y=[]
window_size=500
for i in range(window_size,training_data.shape[1]):
    X.append(training_data[:,i-window_size:i,:])
    Y.append(training_data[:,i,:])
X=np.asarray(X)
X_train=np.asarray(X[:int(0.8*(X.shape[0]))])
Y_train=np.asarray(Y[:int(0.8*(X.shape[0]))])
X_test=np.asarray(X[int(0.8*(X.shape[0])):])
Y_test=np.asarray(Y[int(0.8*(X.shape[0])):])

print("X Shape :", X.shape)
print("X_train Shape :",X_train[:,:,:,0].shape)
print("Y_train Shape :",Y_train[:,:,0].shape)
print("X_test Shape :",X_test[:,:,:,0].shape)
print("Y_test Shape :",Y_test[:,:,0].shape)

plt.figure()
plt.scatter(Demand_scaled,Price_scaled)
plt.title("Demand vs Price ")
print("\n\n","*"*80," Graphs","*"*80)
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
model=keras.Sequential([
    keras.layers.LSTM(units=window_size,return_sequences=True,input_shape=(X_train.shape[1],X_train.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=window_size,return_sequences=True),
    keras.layers.LSTM(units=window_size,return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=window_size),
    
    
    keras.layers.Dense(2,activation='sigmoid')
])
print("*"*80,"Architecture of the Model","*"*80,"\n\n")
model.summary()

model.compile(loss='mse',optimizer='adam')
model.fit(X_train[:,:,:,0],Y_train[:,:,0],epochs=5)
predictions=model.predict(X_train[:,:,:,0])
print(predictions.shape)
plt.plot(predictions[:,0])
plt.plot(Demand_scaled)
plt.figure()
plt.plot(predictions[:,1])
plt.plot(Price_scaled)
# np.sum(np.subtract(predictions,))
from sklearn.metrics import mean_squared_error
mse_demand = mean_squared_error(predictions[:,0],Y_train[:,0,0] )
mse_price = mean_squared_error(predictions[:,1],Y_train[:,1,0] )

print("MSE Demand :",mse_demand)
print("MSE Price :",mse_price)

print(X_train.shape)
print(Y_train.shape)
a=Demand_scaled[-window_size:,0]
b=Price_scaled[-window_size:,0]
c=np.asarray([[a,b]])
print(c.shape)
forecast=400
# print(a[-100:])
for i in range(forecast):
    p=model.predict(np.asarray([[a[-window_size:],b[-window_size:]]]))
    a=np.append(a,p[:,0])
    b=np.append(b,p[:,1])
    

# print(a.shape)
print("Forecast Demand")
plt.plot(a)
plt.figure()
print("Forecast Price")
plt.plot(b)
Y_test.shape
# plt.plot(Y_test[0:800,0,:])
plt.plot(training_data[0,284806:285106,:])
plt.plot(a[window_size:])
plt.legend(['Actual','Forecast'])
plt.title('Demand')
plt.figure()

# plt.plot(Y_test[0:800,1,:])
plt.plot(training_data[1,284406:284806,:])

plt.plot(b[window_size:])
plt.legend(['Actual','Forecast'])
plt.title('Price')

training_data.shape

