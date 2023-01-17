import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/international-airline-passengers/international-airline-passengers.csv", skipfooter = 5)
data.head()
data = data.rename(columns = {"International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60" : "passengers"})
data.info()
data["Month"] = pd.to_datetime(data["Month"])
plt.figure(figsize = (15,8))
plt.plot(data["Month"],data["passengers"])
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.title("Month-Passengers")
plt.show()
data_lstm = data.iloc[:,1].values
data_lstm = data_lstm.reshape(-1,1)
data_lstm = data_lstm.astype("float32")
df = pd.DataFrame(data_lstm)
df.head()
scaler = MinMaxScaler(feature_range = (0, 1)) # scaling
data_lstm = scaler.fit_transform(data_lstm)
train_size = int(len(data_lstm)*0.50)
test_size = len(data_lstm) - train_size
train = data_lstm[0:train_size,:]
test = data_lstm[train_size:len(data_lstm),:]
print("train size: {}, test size: {}".format(len(train),len(test)))
time_step = 2 
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = train[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
trainx = np.array(datax)
trainy = np.array(datay)
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = test[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
testx = np.array(datax)
testy = np.array(datay)
trainx = np.reshape(trainx, (trainx.shape[0], 1 , trainx.shape[1]))
testx = np.reshape(testx, (testx.shape[0], 1 , testx.shape[1]))
model = Sequential()

model.add(SimpleRNN(60, activation = "relu", return_sequences = True, input_shape = (1,time_step)))
model.add(Dropout(0.3))

model.add(SimpleRNN(50, activation = "relu", return_sequences = True))
model.add(Dropout(0.2))

model.add(SimpleRNN(50, activation = "relu", return_sequences = True))
model.add(Dropout(0.2))

model.add(SimpleRNN(40, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = "adam",
             loss = "mean_squared_error",
             metrics = ["accuracy"])

model.summary()
hist = model.fit(trainx,trainy, epochs = 150, batch_size = 32)
trainPredict = model.predict(trainx)
testPredict = model.predict(testx)

trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform([trainy])
testPredict = scaler.inverse_transform(testPredict)
testy = scaler.inverse_transform([testy])

trainScore = math.sqrt(mean_squared_error(trainy[0], trainPredict[:,0]))
print("train score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(testy[0], testPredict[:,0]))
print("test score: %.2f RMSE" % (testScore))

lstm_loss = hist.history["loss"]
plt.figure(figsize = (13,3))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses-Epochs")
plt.grid(True, alpha = 0.5)
plt.plot(lstm_loss)
plt.show()
trainPredictPlot = np.empty_like(data_lstm)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[time_step:len(trainPredict)+time_step, :] = trainPredict

testPredictPlot = np.empty_like(data_lstm)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(time_step*2)+1:len(data_lstm)-1,:] = testPredict

plt.figure(figsize = (13,8))
plt.plot(scaler.inverse_transform(data_lstm),label = "Real Data")
plt.plot(trainPredictPlot,label = "Train Predicted")
plt.plot(testPredictPlot, label = "Test Predicted")
plt.legend()
plt.grid(True, alpha = 0.4)
plt.show()
data_lstm = data.iloc[:,1].values
data_lstm = data_lstm.reshape(-1,1)
data_lstm = data_lstm.astype("float32")
df = pd.DataFrame(data_lstm)
df.head()
scaler = MinMaxScaler(feature_range = (0, 1))
data_lstm = scaler.fit_transform(data_lstm)
time_step = 10 #50
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = train[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
trainx = np.array(datax)
trainy = np.array(datay)
datax = []
datay = []
for i in range(len(test)-time_step-1):
    a = test[i:(i+time_step),0]
    datax.append(a)
    datay.append(test[i + time_step, 0])
testx = np.array(datax)
testy = np.array(datay)
trainx = np.reshape(trainx, (trainx.shape[0], 1 , trainx.shape[1]))
testx = np.reshape(testx, (testx.shape[0], 1 , testx.shape[1]))
model = Sequential()
model.add(LSTM(units = 10, input_shape = (1,time_step)))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", 
              optimizer="adam",
             metrics = ["accuracy"])

model.summary()
hist = model.fit(trainx,trainy, epochs = 40)
trainPredict = model.predict(trainx)
testPredict = model.predict(testx)

trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform([trainy])
testPredict = scaler.inverse_transform(testPredict)
testy = scaler.inverse_transform([testy])

trainScore = math.sqrt(mean_squared_error(trainy[0], trainPredict[:,0]))
print("train score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(testy[0], testPredict[:,0]))
print("test score: %.2f RMSE" % (testScore))

lstm_loss = hist.history["loss"]
plt.figure(figsize = (13,3))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses-Epochs")
plt.grid(True, alpha = 0.5)
plt.plot(lstm_loss)
plt.show()
trainPredictPlot = np.empty_like(data_lstm)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[time_step:len(trainPredict)+time_step, :] = trainPredict

testPredictPlot = np.empty_like(data_lstm)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(time_step*2)+1:len(data_lstm)-1,:] = testPredict

plt.figure(figsize = (13,8))
plt.plot(scaler.inverse_transform(data_lstm),label = "Real Data")
plt.plot(trainPredictPlot,label = "Train Predicted")
plt.plot(testPredictPlot, label = "Test Predicted")
plt.legend()
plt.grid(True, alpha = 0.4)
plt.show()