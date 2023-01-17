#CÉLULA KE-LIB-01

import numpy as np

import keras as K

import tensorflow as tf

import pandas as pd

import seaborn as sns

import os

from matplotlib import pyplot as plt

%matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02

np.random.seed(4)

tf.set_random_seed(13)
#CÉLULA KE-LIB-03

dfBoston = pd.read_csv('../input/boston_mm_tab.csv', header=None)

dfBoston



X = dfBoston[np.arange(0,13)]

y = dfBoston[13]
#CÉLULA KE-LIB-04

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#CÉLULA KE-LIB-05

#Montando a rede neural

tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Inicializador

init = K.initializers.RandomUniform(seed=1)



#Criando o otimizador

simple_sgd = K.optimizers.SGD(lr=0.010)



#Construindo o modelo (topologia)

model = K.models.Sequential()

model.add(K.layers.Dense(units=10, input_dim=13, kernel_initializer=init, activation='tanh')) #1a camada oculta

model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='tanh'))               #2a camada oculta

model.add(K.layers.Dense(units=1,  kernel_initializer=init, activation=None))               #Camada de saída



#Compilando o modelo

model.compile(loss='mean_squared_error', optimizer=simple_sgd, metrics=['mse'])
#CÉLULA KE-LIB-06

#Treinamento

batch_size = 8

max_epochs = 500

print("Iniciando treinamento... ")

h = model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs, shuffle=True, verbose=1)

print("Treinamento finalizado \n")
#CÉLULA KE-LIB-07

#Treinamento

y_pred = model.predict(X_train)

y_d = np.array(y_train).reshape(-1, 1)



results = abs(y_pred - y_d) < np.abs(0.15 * y_d)

results



#Acuracidade

acc = np.sum(results) / len(results)

print("Taxa de acerto do conjunto de treinamento (%): {0:.4f}".format(acc*100) )



#Teste

y_pred = model.predict(X_test)

y_d = np.array(y_test).reshape(-1, 1)



results = abs(y_pred - y_d) < np.abs(0.15 * y_d)

results



#Acuracidade

acc = np.sum(results) / len(results)

print("Taxa de acerto do conjunto de teste (%): {0:.4f}".format(acc*100) )
#CÉLULA KE-LIB-08

# 5 Avaliação do modelo

eval = model.evaluate(X_train, y_train, verbose=0)

print("Erro médio do conjunto de treinamento {0:.4f}".format(eval[0]))



eval = model.evaluate(X_test, y_test, verbose=0)

print("Erro médio do conjunto de teste {0:.4f}".format(eval[0]))
#CÉLULA KE-LIB-9

# Salvando modelo em arquivo

# print("Salvando modelo em arquivo \n")

# mp = "../input/boston_model.h5"

# model.save(mp)
# 7. Usando modelo (operação)

np.set_printoptions(precision=4)

unknown = np.full(shape=(1,13), fill_value=0.6, dtype=np.float32)

unknown[0][3] = -1.0 # encodando o booleano

predicted = model.predict(unknown)

print("Usando o modelo para previsão de preço médio de casa para as caracteristicas: ")

print(unknown)

print("\nO preço médio será [dolares]: ")

print(predicted * 10000)