# Importando bibliotecas

import numpy

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

numpy.random.seed(7)
# Carrega apenas a coluna com o total de passageiros por mês, em milhares (112 = 122000 passageiros em vôos)

dataframe = pandas.read_csv('../input/airlines-passenger-data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

dataframe.head(3)
dataframe.plot()
#Converte a coluna do dataframe pandas em um vetor numpy

dataset = dataframe.values

dataset = dataset.astype('float32')



# Normaliza os dados para ficarem entre 0 e 1

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)



# Divite os dados de treino (2/3) e teste (1/3)

# Note que a divisão não é aleatória, mas sim sequencial

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# Recebe uma série e converte em uma matriz com séries deslocadas.

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1

look_back = 12

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

# shape is [samples, time steps, features]



trainX = trainX.reshape(-1, look_back, 1)

testX = testX.reshape(-1, look_back, 1)



trainX.shape, testX.shape
for k in zip(trainX[:5], trainY[:5]):

    print(k[0],k[1])
model = Sequential()

model.add(LSTM(8, input_shape=(look_back, 1), return_sequences=True))

model.add(LSTM(8))

model.add(Dense(1)) 

model.compile(loss='mean_squared_error', optimizer='rmsprop')

model.summary()
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=0)



# Realiza as previsões. Notar que a utilidade de prever trainX é nenhuma. Serve apenas para exibir no gráfico.

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
# Reescala os números para os valores originais (milhares de passageiros)

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])
# Calcula os erros de previsão

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting

trainPredictPlot = numpy.empty_like(dataset)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict



# shift test predictions for plotting

testPredictPlot = numpy.empty_like(dataset)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict



# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()