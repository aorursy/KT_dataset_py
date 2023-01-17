import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

import sklearn.metrics as mt

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.layers import LSTM

df = pd.read_csv("../input/air-traffic-passenger-statistics.csv")

df.head()
df.loc[:,"Activity Period"] = pd.to_datetime(df.loc[:,"Activity Period"].astype(str), format="%Y%m")

df.loc[:,"Year"] = df["Activity Period"].dt.year

df.loc[:,"Month"] = df["Activity Period"].dt.month
df.columns
df.isna().sum()
for i in ['GEO Summary','GEO Region','Activity Type Code','Price Category Code',

          'Terminal','Boarding Area','Month']:

    A = df[i].value_counts().index.tolist()

    Vals = list(pd.value_counts(df[i]))

    explode = [0.1]*len(A)

    plt.figure(figsize=(10,10))

    plt.title(i)

    plt.pie(Vals,explode=explode,labels=A,autopct='%.1f%%')

    plt.show()
sns.pairplot(df.drop(['Operating Airline IATA Code','Published Airline IATA Code'],axis=1))
for i in ['GEO Summary','GEO Region','Activity Type Code','Price Category Code',

          'Terminal','Boarding Area','Month']:

    sns.pairplot(hue=i,data=df.drop(['Operating Airline IATA Code','Published Airline IATA Code'],axis=1))
spivot = pd.pivot_table(df, index='Month', columns = 'Year', values = 'Passenger Count', aggfunc=np.mean)

spivot.plot(figsize=(20,10), linewidth=3)

plt.show()
total = float(len(df))

for i in ['Operating Airline','Published Airline']:

    plt.figure(figsize=(20,8))

    plt.xticks(rotation=90)

    ax = sns.countplot(x=i, data=df)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha="center") 

plt.show()
total = float(len(df))

for i in ['Operating Airline','Published Airline']:

    for j in ['Price Category Code','GEO Summary']:

        plt.figure(figsize=(20,8))

        plt.xticks(rotation=90)

        ax = sns.countplot(x=i,hue=j,data=df)

        for p in ax.patches:

            height = p.get_height()

            ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format((height/total)*100),ha="center") 

    plt.show()
df[['Activity Period','Passenger Count']].iplot(kind='box')
TS = df.groupby("Activity Period")["Passenger Count"].sum().to_frame()
TS.iplot()
s = MinMaxScaler(feature_range=(0,1))

TS = s.fit_transform(TS)


train_size = int(len(TS) * 0.66)

test_size = len(TS) - train_size

train, test = TS[0:train_size,:], TS[train_size:len(TS),:]

print(len(train), len(test))
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

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(trainX, trainY, epochs=100, batch_size=2,validation_data=(testX,testY), verbose=2)
plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='test')

plt.legend()

plt.show()
trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

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

print('Test Score: %.2f RMSE' % (testScore))
trainPredictPlot = np.empty_like(TS)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = np.empty_like(TS)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(TS)-1, :] = testPredict

# plot baseline and predictions

plt.plot(s.inverse_transform(TS))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()