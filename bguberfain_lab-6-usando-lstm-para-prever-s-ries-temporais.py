# Importando bibliotecas

import numpy

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM, Conv1D, Dropout

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

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



look_back = 12



# Divite os dados de treino (2/3) e teste (1/3)

# Note que a divisão não é aleatória, mas sim sequencial

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size-look_back-1:len(dataset),:]
# Recebe uma série e converte em uma matriz com séries deslocadas.

def create_dataset(dataset, look_back=1, std=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back):

        a = dataset[i:(i+look_back), 0]-dataset[i, 0]

        a /= std

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0]-dataset[i + look_back-1, 0])

    return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1

std = train[:, 0].std()

trainX, trainY = create_dataset(train, look_back, std)

testX, testY = create_dataset(test, look_back, std)

# shape is [samples, time steps, features]



trainX = trainX.reshape(-1, look_back, 1)

testX = testX.reshape(-1, look_back, 1)

trainY = trainY / 30

testY = testY / 30



trainX.shape, testX.shape
model = Sequential()

model.add(LSTM(32, input_shape=(look_back, 1), return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
callbacks = [

    ReduceLROnPlateau(patience=10, factor=0.5, verbose=True),

    ModelCheckpoint('best.model', save_best_only=True),

    EarlyStopping(patience=25, verbose=True)

]



history = model.fit(trainX, trainY, epochs=5000, batch_size=24, validation_data=(testX, testY),

                    verbose=0, callbacks=callbacks)
df_history = pandas.DataFrame(history.history)

ax = df_history[['val_loss', 'loss']].plot(figsize=(10, 5))

df_history['lr'].plot(ax=ax.twinx(), color='gray')
# Realiza as previsões. Notar que a utilidade de prever trainX é nenhuma. Serve apenas para exibir no gráfico.

model.load_weights('best.model')

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
plt.scatter(trainY, trainPredict.ravel())
plt.scatter(testY, testPredict.ravel())
# Calcula os erros de previsão

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting

trainPredictPlot = (trainPredict.ravel() * 30) + dataset[look_back:len(trainPredict)+look_back, 0]



# shift test predictions for plotting

testPredictPlot = (testPredict.ravel() * 30) + dataset[len(trainPredict)+(look_back)-1:len(dataset), 0]



# plot baseline and predictions

plt.figure(figsize=(20, 10))

plt.plot(dataset)

plt.plot(look_back+numpy.arange(len(trainPredictPlot)), trainPredictPlot)

plt.plot(look_back+numpy.arange(len(testPredictPlot))+len(trainPredictPlot)-1, testPredictPlot)

plt.show()