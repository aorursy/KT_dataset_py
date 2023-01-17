import numpy

import matplotlib.pyplot as plt

import pandas as pd

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv", parse_dates=['date'])

#test=pd.read_csv("../input/test.csv",parse_dates=['date'])

stores = pd.read_csv("../input/stores.csv") 
t=train.groupby([ 'store_nbr','date'], as_index=False).agg({"unit_sales": "sum"})

train = pd.merge(t, stores, how='left', on=['store_nbr'])

mask=train['state']=='Pichincha'

train=train.loc[mask]

train=train.groupby(['date'], as_index=False).agg({"unit_sales": "sum"})

train.head(10)
#train1.groupby('month', as_index=False).agg({"unit_sales": "sum"})



#MUPI_COM["valor"].plot()
train.tail(12)


plt.plot(train['unit_sales'])

plt.show()
numpy.random.seed(123)
train_size = int(len(train) * 0.75)

test_size = len(train) - train_size



print(train_size,test_size, len(train))
train1= train[0:train_size]

test =  train[train_size:len(train)]

print(len(train1), len(test))
train1=train1.set_index("date")

test=test.set_index("date")

train=train.set_index("date")

train1=train1.values

test=test.values

train=train.values
def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)



look_back = 1

trainX, trainY = create_dataset(train1, look_back)

testX, testY = create_dataset(test, look_back)


model = Sequential()

model.add(Dense(8, input_dim=look_back, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)
trainScore = model.evaluate(trainX, trainY, verbose=0)

print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(testX, testY, verbose=0)

print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

 

trainPredictPlot = numpy.empty_like(train)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

 

testPredictPlot = numpy.empty_like(train)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(train)-1, :] = testPredict

 

plt.plot(train)

plt.show()

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()

plt.plot(train)

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()


