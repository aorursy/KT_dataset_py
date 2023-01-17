import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

data=pd.read_csv('../input/Google.csv')
data=data.iloc[:,4].values

data=pd.DataFrame(data)



len(train_data)
train_data=data.iloc[0:2500].values

test_data=data.iloc[2500: ,].values

test_data=pd.DataFrame(test_data)

train_data=pd.DataFrame(train_data)

from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler(feature_range =(0,1))

train_data_scaled=sc.fit_transform(train_data)

train_data_scaled=pd.DataFrame(train_data_scaled)
x_train=[]

y_train=[]
for i in range(90,2500):

    x_train.append(train_data_scaled.iloc[i-90:i,0])

    y_train.append(train_data_scaled.iloc[i,0])
x_train,y_train=np.array(x_train), np.array(y_train) 

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1 ))   

    
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
regressor=Sequential()

regressor.add(LSTM(units = 50,return_sequences = True,input_shape=(x_train.shape[1],1)))

regressor.add(Dropout(0.2))   

regressor.add(LSTM(units = 50,return_sequences = True,))

regressor.add(Dropout(0.2))      
regressor.add(LSTM(units = 50,return_sequences = True,))

regressor.add(Dropout(0.2))     
regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))    
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam',loss='mean_squared_error')
regressor.fit(x_train,y_train,epochs=50,batch_size=32)
data.head()
print(data)
input=data.iloc[len(data)-len(test_data)-90: ].values



input=pd.DataFrame(input)
type(input)
input=sc.transform(input)




x_test=[]



for i in range(90,len(input)):

    x_test.append(input.iloc[i-90:i, 0])

        

x_test=np.array(x_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)

predicted_stock_price=sc.inverse_transform(predicted_stock_price)
pred=pd.DataFrame(predicted_stock_price)

plt.plot(test_data,color='red',label='real stock price')

plt.plot(predicted_stock_price,color='blue',label='predicted stock price')

plt.title('goole stock price')

plt.xlabel('time')

plt.ylabel('price')

plt.legend()

plt.show()
import math

from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(test_data, predicted_stock_price))
print(rmse)