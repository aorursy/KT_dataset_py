#CÉLULA KE-LIB-01

import numpy as np

import keras as K

import tensorflow as tf

import pandas as pd

import os

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02

np.random.seed(1)

tf.set_random_seed(1)
#CÉLULA KE-LIB-03

#IMPLEMENTE O CÓDIGO AQUI 

dfPetr4 = pd.read_csv('../input/petr4.csv')
#CÉLULA KE-LIB-04

#IMPLEMENTE O CÓDIGO AQUI 

dfPetr4 = dfPetr4.dropna()
#CÉLULA KE-LIB-05

#IMPLEMENTE O CÓDIGO AQUI 

openValues = dfPetr4['Open']

openValues = openValues.values.reshape(-1,1)





plt.figure(figsize=(12,8))

plt.plot(openValues)
#CÉLULA KE-LIB-06

#IMPLEMENTE O CÓDIGO AQUI (1 linha)

from sklearn.preprocessing import MinMaxScaler
#CÉLULA KE-LIB-07

#IMPLEMENTE O CÓDIGO AQUI

scaler = MinMaxScaler(feature_range=(-1,1))

scaler = MinMaxScaler(feature_range=(-1,1))

openValuesScaled = MinMaxScaler(feature_range=(-1,1))
#CÉLULA KE-LIB-08

from sklearn.model_selection import train_test_split

time_steps = 30

prediction = 1 #valor a ser previsto (um dia a frente)

num_features = 1 # numero de atributos usados
#CÉLULA KE-LIB-09

#Criação de conjunto de treinamento, utilizando o subitem 1 da descrição acima

#As variáveis de treinamento e teste deverão ser: X_train, X_test, y_train e y_test

#Se você fez a normalização usando MinMaxScaler, user a variável openValuesScaled. Caso contrario use a variavel openValues

#Este trecho de código abaixo, faz a montagem do dataset de treinamento conforme descrição do subitem 1



#A variável time steps corresponde a 30 dias da sequencia, e a variavel num_features corresponde a quantidade de atributos

#da entrada (1 valor, somente o campo 'Open')



X = openValues[0:(len(openValues) - (len(openValues) % time_steps))]

X_train = X.reshape(-1, time_steps, num_features) #Esta formatação é necessária para o LSTM



#Ao realizar a formatação da linha acima, a variavel X_train será um tensor de dimensões (41,30,1).

#Neste caso não foram utilizados todos os valores do dataset de modo que será possível construir um conjunto adicional 

# de validação, conforme o código abaixo



#Valores de Saída: aqui a saída é composta por apenas 1 valor (1 label previsto após a exibição da sequencia de 30 dias)

y_train = X_train[1:, 0, 0]

y_train  = np.append(y_train, openValues[1230])

y_train = y_train.reshape(-1, 1)

print(y_train.shape)



#Conjunto de validação com os dados remanescentes do dataset

remain = len(openValues) - len(X)

X_aux = openValues[ - remain - time_steps :]



#o -1 é para não incluir a ultima linha porque nao sabemos o valor de saída neste caso

y_val = []

y_test_val = []

for start in range(len(X_aux) - time_steps + prediction - 1) : 

    row = X_aux[start:time_steps + start]

    y_val.append(row)

    y_test_val.append(X_aux[time_steps + start])



y_val = np.array(y_val)

y_test_val = np.array(y_test_val)



#AS variáveis X_train e y_train contém os dados uteis formatados conforme o subitem 1. Vamos dividir parte deste array para testes

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0, shuffle=False)
#CÉLULA KE-LIB-10

#Montando a rede recorrente 'LSTM-Dense'

RS_VALUE = False

batch_size = 1



#Limpa o modelo previamente usado, senão irá acumular camadas

K.backend.clear_session()



tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



init   = K.initializers.glorot_uniform(seed=1)



#IMPLEMENTE O CÓDIGO AQUI



theOptimizer = K.optimizers.Adam()



model = K.models.Sequential()

model.add(K.layers.LSTM(units=15, return_sequences=RS_VALUE, batch_input_shape=(batch_size, time_steps, prediction), kernel_initializer=init, dropout=0.2, activation='relu'))

model.add(K.layers.Dense(units=1,  kernel_initializer=init, activation=None))



# Compile o modelo - IMPLEMENTE O CÓDIGO AQUI

model.compile(loss='mean_squared_error', optimizer=theOptimizer, metrics=['mae', 'mse'])

model.summary()
#CÉLULA KE-LIB-11

#Treinamento

#IMPLEMENTE O CÓDIGO AQUI

max_epochs = 100

print("Iniciando treinamento... ")

h = model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs, shuffle=True, verbose=True)

print("Treinamento finalizado \n")
#CÉLULA KE-LIB-12

# Avaliação do modelo

#IMPLEMENTE O CÓDIGO AQUI

#eval = model.evaluate(X_train, y_train, verbose=False)

#print("Dados de teste: loss = %0.6f MAE = %0.2f%% \n" \

#% (eval[0], eval[1]*100) )
#CÉLULA KE-LIB-13

# Salvando modelo em arquivo

#print("Salvando modelo em arquivo \n")

#mp = ".\\stock_model.h5"

#model.save(mp)
#CÉLULA KE-LIB-14

#Este trecho de código realiza a previsão dos valores de teste e plota o gráfico correspondente

y_pred_list = []

for i in range(len(X_test)) :

    X_data = X_test[i].reshape(1,30,1)

    y_pred = model.predict(X_data)

    y_pred_list.append(y_pred)

    

y_p = np.array(y_pred_list).ravel() #este comando faz o 'flatten' do array para que possa ser colocado no gráfico

y_t = np.array(y_test).ravel()



plt.figure(figsize=(12,8))

plt.plot(y_p, 'b')

plt.plot(y_t, 'r')
#CÉLULA KE-LIB-15

#IMPLEMENTE O CÓDIGO AQUI

plt.plot(h.history['loss'])
#CÉLULA KE-LIB-16

# Este trecho de código realiza o teste com o conjunto de validação 

y_pred_list = []

for i in range(len(y_val)) :

    y_data = y_val[i].reshape(1,30,1)

    y_pred = model.predict(y_data)

    y_pred_list.append(y_pred)



y_p = np.array(y_pred_list).ravel()

y_t = np.array(y_test_val).ravel()

plt.figure(figsize=(12,8))

plt.plot(y_p, 'b')

plt.plot(y_t, 'r')    