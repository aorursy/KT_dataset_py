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

import matplotlib.pyplot as plt

import numpy as np 

import seaborn as sns



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM

import tensorflow as tf





import math

from sklearn.metrics import mean_squared_error



from numpy import array
df=pd.read_csv("/kaggle/input/icici-bank-date/ICICIBANK.csv.csv")
df.head(10)
df.tail(20)
len(df)
df.isnull().sum()
datadrop=df.dropna()
datadrop
datadrop.isnull().sum()
df1=datadrop.reset_index()['Close']

df1
len(df1)
figsize=(30,50)

plt.plot(df1)
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
df1
##splitting dataset into train and test split
training_size=int(len(df1)*0.65)

test_size=len(df1)-training_size

train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
training_size,test_size
train_data
def create_datasets(dataset,time_step=1):

    dataX,dataY=[], []

    for i in range(len(dataset)-time_step-1):

        a=dataset[i:(i+time_step),0]

        dataX.append(a)

        dataY.append(dataset[i+time_step,0])

    return np.array(dataX) , np.array(dataY)
time_step=100

X_train, y_train = create_datasets(train_data ,time_step)

X_test , y_test  = create_datasets(test_data , time_step)
print ("TRAINING DATASETS DATA ","X")

print(X_train)

print (len(X_train))



print ("TEST  DATASETS DATA ","X")

print(X_test)

print (len(X_test))
print ("TRAINING DATASETS DATA ","Y")

print(y_train)

print (len(y_train))



print ("TEST  DATASETS DATA ","Y")

print(y_test)

print (len(y_test))
print(X_train.shape), print(y_train.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1], 1)

X_test=X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
## Create the Stacked LSTM model
model=Sequential()

model.add(LSTM(50, return_sequences=True , input_shape=(100,1)))

model.add(LSTM(50, return_sequences=True))

model.add(LSTM(50))

model.add(Dense(1))

model.compile(loss="mean_squared_error" , optimizer="adam")
model.summary()
model.summary()
model.fit(X_train,y_train, validation_data=(X_test,y_test),epochs=100, batch_size=64,verbose=1)
tf.__version__
## PREDICTION
train_predict=model.predict(X_train)

test_predict=model.predict(X_test)
train_predict=scaler.inverse_transform(train_predict)

test_predict=scaler.inverse_transform(test_predict)
math.sqrt(mean_squared_error(y_train,train_predict))
### Test Data RMSE

math.sqrt(mean_squared_error(y_test,test_predict))
## PLOT THE TEST DATA ON THE WHOLE DATASET , TO CHECK OUR MACHINE LEARNIG 
##TRAIN PREDICITONS FOR PLOTTING

look_back=100

trainPredictPlot=np.empty_like(df1)

trainPredictPlot[: ,:] =np.nan

trainPredictPlot[look_back:len(train_predict)+look_back , :]=train_predict
testPredictPlot=np.empty_like(df1)

testPredictPlot[:,:]=np.nan

testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :]=test_predict
plt.plot(scaler.inverse_transform(df1))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

x_input=test_data[1669:].reshape(1,-1)

x_input.shape
temp_input=list(x_input)

temp_input=temp_input[0].tolist()
temp_input
## NEXT #) DAYS
len(test_data)
from numpy import array



lst_output=[]

n_steps=100

i=0

while(i<30):

    

    if(len(temp_input)>100):

        #print(temp_input)

        x_input=np.array(temp_input[1:])

        print("{} day input {}".format(i,x_input))

        x_input=x_input.reshape(1,-1)

        x_input = x_input.reshape((1, n_steps, 1))

        #print(x_input)

        yhat = model.predict(x_input, verbose=0)

        print("{} day output {}".format(i,yhat))

        temp_input.extend(yhat[0].tolist())

        temp_input=temp_input[1:]

        #print(temp_input)

        lst_output.extend(yhat.tolist())

        i=i+1

    else:

        x_input = x_input.reshape((1, n_steps,1))

        yhat = model.predict(x_input, verbose=0)

        print(yhat[0])

        temp_input.extend(yhat[0].tolist())

        print(len(temp_input))

        lst_output.extend(yhat.tolist())

        i=i+1

    



print(lst_output)
day_new=np.arange(1,101)

day_pred=np.arange(101,131)
day_new
len(df1)
plt.plot(day_new,scaler.inverse_transform(df1[4952:]))

plt.plot(day_pred,scaler.inverse_transform(lst_output))
df3=df1.tolist()

df3.extend(lst_output)

plt.plot(df3[5000:])
df3=scaler.inverse_transform(df3).tolist()
df3
plt.plot(df3)

# Visualize the data

plt.title('Model')

plt.xlabel('Date', fontsize=8)

plt.ylabel('Close Price USD', fontsize=6)

plt.show()

df5=df1.tolist()

df5.extend(lst_output)

plt.plot(df5[4800:])
df5