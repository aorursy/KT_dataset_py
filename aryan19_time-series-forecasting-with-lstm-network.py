import numpy as np

import pandas as pd

from pandas.plotting import autocorrelation_plot as acp

import matplotlib.pyplot as plt

import plotly_express as px

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as mt

import math

import keras

from keras.layers import Dense,LSTM,Dropout

from keras.models import Sequential

df = pd.read_csv("../input/portland-oregon-average-monthly-.csv")

df.head()
df.columns = ['Month','Avg Ridership']
df.info()
df['Avg Ridership'].unique()
df['Avg Ridership'] = df['Avg Ridership'].replace(' n=114',np.nan)

df = df.dropna()
df = df.dropna()

df['Avg Ridership'].unique()
df['Avg Ridership'] = pd.to_numeric(df['Avg Ridership'])
df.info()
px.line(df,x='Month',y='Avg Ridership').show()
df = df.set_index('Month')
s = MinMaxScaler(feature_range=(0,1))

DF = s.fit_transform(df)
train_size = int(len(DF) * 0.66)

test_size = len(DF) - train_size

train, test = DF[0:train_size,:], DF[train_size:len(DF),:]

print(f'Training Size = {len(train)}, Testing Size = {len(test)}')
def create_dataset(S, look_back=1):

    dataX, dataY = [], []

    for i in range(len(S)-look_back-1):

        a = S[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(S[i + look_back, 0])

    return np.array(dataX), np.array(dataY)
look_back = 1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()

model.add(LSTM(128, input_shape=(1, look_back)))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=100, batch_size=2,validation_data=(testX,testY), verbose=2)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
trainPredict = model.predict([trainX])

testPredict = model.predict([testX])

#Changing prediction to it's original units

trainPredict = s.inverse_transform(trainPredict)

trainY = s.inverse_transform([trainY])

testPredict = s.inverse_transform(testPredict)

testY = s.inverse_transform([testY])



trainScore = math.sqrt(mt.mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score = %.2f MSE' % mt.mean_squared_error(trainY[0],trainPredict[:,0]))

print('Train Score =  %.2f RMSE' % (trainScore))

testScore = math.sqrt(mt.mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score = %.2f MSE' % mt.mean_squared_error(testY[0],testPredict[:,0]))

print('Test Score = %.2f RMSE' % (testScore))
trainPredictPlot = np.empty_like(DF)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = np.empty_like(DF)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(DF)-1, :] = testPredict

# plot baseline and predictions

plt.plot(s.inverse_transform(DF))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()