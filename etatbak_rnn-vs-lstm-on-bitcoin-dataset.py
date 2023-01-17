# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

warnings.filterwarnings("ignore")





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
bit_data=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv")

bit_data["date"]=pd.to_datetime(bit_data["Timestamp"],unit="s").dt.date

group=bit_data.groupby("date")

data=group["Close"].mean()
data.shape
data.isnull().sum()
data.head()
close_train=data.iloc[:len(data)-50]

close_test=data.iloc[len(close_train):]
#feature scalling (set values between 0-1)

close_train=np.array(close_train)

close_train=close_train.reshape(close_train.shape[0],1)

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

close_scaled=scaler.fit_transform(close_train)
timestep=50

x_train=[]

y_train=[]



for i in range(timestep,close_scaled.shape[0]):

    x_train.append(close_scaled[i-timestep:i,0])

    y_train.append(close_scaled[i,0])



x_train,y_train=np.array(x_train),np.array(y_train)

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1) #reshaped for RNN

print("x_train shape= ",x_train.shape)

print("y_train shape= ",y_train.shape)
from keras.models import Sequential

from keras.layers import Dense, SimpleRNN, Dropout,Flatten



regressor=Sequential()

#first RNN layer

regressor.add(SimpleRNN(128,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))

regressor.add(Dropout(0.25))

#second RNN layer

regressor.add(SimpleRNN(256,activation="relu",return_sequences=True))

regressor.add(Dropout(0.25))

#third RNN layer

regressor.add(SimpleRNN(512,activation="relu",return_sequences=True))

regressor.add(Dropout(0.35))

#fourth RNN layer

regressor.add(SimpleRNN(256,activation="relu",return_sequences=True))

regressor.add(Dropout(0.25))

#fifth RNN layer

regressor.add(SimpleRNN(128,activation="relu",return_sequences=True))

regressor.add(Dropout(0.25))

#convert the matrix to 1-line

regressor.add(Flatten())

#output layer

regressor.add(Dense(1))



regressor.compile(optimizer="adam",loss="mean_squared_error")

regressor.fit(x_train,y_train,epochs=100,batch_size=64)
inputs=data[len(data)-len(close_test)-timestep:]

inputs=inputs.values.reshape(-1,1)

inputs=scaler.transform(inputs)
x_test=[]

for i in range(timestep,inputs.shape[0]):

    x_test.append(inputs[i-timestep:i,0])

x_test=np.array(x_test)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
predicted_data=regressor.predict(x_test)

predicted_data=scaler.inverse_transform(predicted_data)
data_test=np.array(close_test)

data_test=data_test.reshape(len(data_test),1)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(data_test,color="r",label="true result")

plt.plot(predicted_data,color="b",label="predicted result")

plt.legend()

plt.xlabel("Time(50 days)")

plt.ylabel("Close Values")

plt.grid(True)

plt.show()
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout,Flatten



model=Sequential()



model.add(LSTM(10,input_shape=(None,1),activation="relu"))



model.add(Dense(1))



model.compile(loss="mean_squared_error",optimizer="adam")



model.fit(x_train,y_train,epochs=100,batch_size=32)
inputs=data[len(data)-len(close_test)-timestep:]

inputs=inputs.values.reshape(-1,1)

inputs=scaler.transform(inputs)
x_test=[]

for i in range(timestep,inputs.shape[0]):

    x_test.append(inputs[i-timestep:i,0])

x_test=np.array(x_test)

x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
predicted_data=model.predict(x_test)

predicted_data=scaler.inverse_transform(predicted_data)
data_test=np.array(close_test)

data_test=data_test.reshape(len(data_test),1)
plt.figure(figsize=(8,4), dpi=80, facecolor='w', edgecolor='k')

plt.plot(data_test,color="r",label="true result")

plt.plot(predicted_data,color="b",label="predicted result")

plt.legend()

plt.xlabel("Time(50 days)")

plt.ylabel("Close Values")

plt.grid(True)

plt.show()