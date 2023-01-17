import numpy as np # Linear algebra.

import pandas as pd # Data processing.

import matplotlib.pyplot as plt # Visualize

import math

from keras.models import Sequential # Create Model

from keras.layers import Dense # Neurons

from keras.layers import LSTM # Long Short Term Memory

from sklearn.preprocessing import MinMaxScaler # Normalize

from sklearn.metrics import mean_squared_error # Loss Function

from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/Tesla.csv - Tesla.csv.csv") # Import data

data.head()
df = data.iloc[:,1].values # We use "Open" column.

plt.plot(df)

plt.xlabel("Time")

plt.ylabel("Open")

plt.show()
df = df.reshape(-1,1)



scaler = MinMaxScaler(feature_range = (0,1)) # Normalize data

df = scaler.fit_transform(df)

np.max(df)
# Test - Train Split

train_size = int(len(df) * 0.75) # % 75 Train

test_size = len(df) - train_size # % 25 Test

print("Train Size :",train_size,"Test Size :",test_size)



train = df[0:train_size,:]

test = df[train_size:len(df),:]
time_stemp = 10



datax = []

datay = []

for i in range(len(train)-time_stemp-1):

    a = train[i:(i+time_stemp), 0]

    datax.append(a)

    datay.append(train[i + time_stemp, 0])

trainx = np.array(datax)

trainy = np.array(datay)





datax = []

datay = []

for i in range(len(test)-time_stemp-1):

    a = test[i:(i+time_stemp), 0]

    datax.append(a)

    datay.append(test[i + time_stemp, 0])

testx = np.array(datax)

testy = np.array(datay)



trainx = np.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1])) # For Keras

testx = np.reshape(testx, (testx.shape[0], 1,testx.shape[1])) # For Keras

print(trainx.shape)

testx.shape
epochs = 200

model = Sequential()

model.add(LSTM(10, input_shape = (1, time_stemp)))

model.add(Dense(1)) # Output Layer

model.compile(loss = "mean_squared_error", optimizer = "adam")

history = model.fit(trainx,trainy, epochs = epochs, batch_size = 50, verbose=0)

# As you can see, Loss is very little
epoch = np.arange(0, epochs, 10)

losses = []

for i in epoch:

    if i % 10 == 0:

        losses.append(history.history["loss"][i])

        

data = {"epoch":epoch,"loss":losses}

data = pd.DataFrame(data) # Create dataframe for visualize with plotly



# Visualize

import plotly.express as px



fig = px.line(data,x="epoch",y="loss",width = 1200, height = 500)

fig.show()

# I choose plotly for visualize because it's interactive
train_predict = model.predict(trainx)

test_predict = model.predict(testx)



train_predict = scaler.inverse_transform(train_predict)

test_predict = scaler.inverse_transform(test_predict)

trainy = scaler.inverse_transform([trainy])

testy = scaler.inverse_transform([testy])



train_score = math.sqrt(mean_squared_error(trainy[0], train_predict[:,0])) # mean_squared_error -> Loss Function

print("Train Score : %2.f RMSE" % (train_score))

test_score = math.sqrt(mean_squared_error(testy[0], test_predict[:,0]))

print("Test Score : %2.f RMSE" % (test_score))
# empty_like -> Return a new array with the same shape and type as a given array.

train_predict_plot = np.empty_like(df)

train_predict_plot[:,:] = np.nan

train_predict_plot[time_stemp:len(train_predict)+time_stemp, :] = train_predict



test_predict_plot = np.empty_like(df)

test_predict_plot[:, :] = np.nan

test_predict_plot[len(train_predict)+(time_stemp*2)+1:len(df)-1, :] = test_predict



plt.plot(scaler.inverse_transform(df),color = "red",label = "Real")

plt.plot(train_predict_plot,label = "Train Predict",color = "yellow",alpha = 0.7)

plt.plot(test_predict_plot,label = "Test Predict",color = "green", alpha = 0.7)

plt.legend()

plt.xlabel("Time")

plt.ylabel("Open Value")

plt.show()