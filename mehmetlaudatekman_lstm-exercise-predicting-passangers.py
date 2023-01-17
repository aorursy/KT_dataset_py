# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

sns.set_style('darkgrid')

from matplotlib import pyplot as plt





import math

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/international-airline-passengers/international-airline-passengers.csv',skipfooter=5)

passanger_values = data.iloc[:,1].values
data.head()
fig,ax = plt.subplots(figsize=(10,7))

plt.plot(passanger_values,color="green")

plt.xlabel("Time")

plt.ylabel("Passanger Count (Thousand)")

plt.show()
passanger_values = passanger_values.reshape(-1,1) # Converting scalar into vector

passanger_values = passanger_values.astype("float32") # Changing Type 

passanger_values.shape
scaler = MinMaxScaler(feature_range=(0,1)) # Scaling

passanger_values = scaler.fit_transform(passanger_values)

passanger_values[:5]
train = []

test = []



train_len = 71

test_len = 71



train = passanger_values[:71]

test = passanger_values[71:]



print("The length of train",len(train))

print("The length of test",len(test))
time_stemp = 10

dataX = []

dataY = []

for i in range(len(train)-time_stemp-1):

    a = train[i:(i+time_stemp), 0]

    dataX.append(a)

    dataY.append(train[i + time_stemp, 0])

trainX = np.array(dataX)

trainY = np.array(dataY)  

print("The shape of trainX",trainX.shape)

print("The shape of trainY",trainY.shape)
time_step = 10

dataX,dataY = [],[]

for i in range(len(train)-time_step-1):

    a = test[i:(i+time_step), 0]

    dataX.append(a)

    dataY.append(test[i + time_step, 0])

testX = np.array(dataX)

testY = np.array(dataY)

print("The shape of testX",testX.shape)

print("The shape of testY",testY.shape)
testX = testX.reshape((60,1,10))

trainX = trainX.reshape((60,1,10))

print("The shape of testX",testX.shape)

print("The shape of trainX",trainX.shape)
from keras.layers import Dense,Dropout,LSTM

from keras.models import Sequential



regressor = Sequential()

regressor.add(LSTM(10,input_shape = (1,10)))

regressor.add(Dense(1))

regressor.compile(optimizer = "adam",loss="mean_squared_error")
regressor.fit(trainX,trainY,epochs=50)
#Predicting

trainPred =regressor.predict(trainX) 

testPred = regressor.predict(testX)



#Taking Inverse

trainPred = scaler.inverse_transform(trainPred)

testPred = scaler.inverse_transform(testPred)



trainY = scaler.inverse_transform([trainY])

testY = scaler.inverse_transform([testY])



#Computing RMSE (Root Mean Squared Error)

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPred[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPred[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shifting train

trainPredictPlot = np.empty_like(passanger_values)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[time_step:len(trainPred)+time_step, :] = trainPred





# shifting test predictions for plotting

testPredictPlot = np.empty_like(passanger_values)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPred)+(time_step*2)+1:len(passanger_values)-1, :] = testPred





# plot baseline and predictions

fig,ax = plt.subplots(figsize=(10,7))

plt.plot(scaler.inverse_transform(passanger_values))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()