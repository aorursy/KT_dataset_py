#CÉLULA KE-LIB-01

import numpy as np

import keras as K

import tensorflow as tf

import pandas as pd

import seaborn as sns

import os

from keras.utils import to_categorical

import h5py

from matplotlib import pyplot as plt

%matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02

def loadDatasets():

    train_dataset = h5py.File('../input/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # features de treinamento

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # labels de treinamento



    test_dataset = h5py.File('../input/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # features de teste

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # labels de teste



    classes = np.array(test_dataset["list_classes"][:]) # lista das classes



    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))



    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
#IMPLEMENTE O CÓDIGO AQUI (1 linha)

X_train, y_train, X_test, y_test, classes = loadDatasets()
#IMPLEMENTE O CÓDIGO AQUI (1 linha)

X_train[0]
#IMPLEMENTE O CÓDIGO AQUI

index = 10

plt.figure()

plt.imshow(X_train[index])
#CÉLULA KE-LIB-03

np.random.seed(4)

tf.set_random_seed(13)



#IMPLEMENTE O CÓDIGO AQUI (2 linhas)

X_train = X_train/255

X_test  = X_test/255
#CÉLULA KE-LIB-04

#IMPLEMENTE O CÓDIGO AQUI (2 linhas)

X_train_flatten = X_train.reshape(X_train.shape[0], -1)

X_test_flatten  = X_test.reshape(X_test.shape[0], -1)
#IMPLEMENTE O CÓDIGO AQUI (2 linhas)

X_train_flatten.shape
#CÉLULA KE-LIB-05

layers_dims = [12288, 20, 1] # modelo de 2 camadas densas



tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Limpa o modelo previamente usado, senão irá acumular camadas

K.backend.clear_session()



#Inicializador

init = K.initializers.glorot_uniform()



#IMPLEMENTE O CÓDIGO AQUI



#Criando o otimizador

theOptimizer = K.optimizers.RMSprop()



#Construindo o modelo (topologia)

model = K.models.Sequential()



#Exemplo de input_shape na camada de entrada da rede

# model = Sequential()

# model.add(Dense(32, input_shape=(16,)))

# o modelo irá receber como entrada arrays no formato (*, 16)

# e como saída terá arrays no formato (*, 32)



#Rede Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=20, input_dim=12288, kernel_initializer=init, activation='relu'))

model.add(K.layers.Dense(units=1,kernel_initializer=init, activation='sigmoid'))



# Compile o modelo - IMPLEMENTE O CÓDIGO AQUI    

model.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['acc'])

model.summary()
#CÉLULA KE-LIB-06

#IMPLEMENTE O CÓDIGO AQUI

batch_size = 19

max_epochs = 2500 



print("Iniciando treinamento... ")

#IMPLEMENTE O CÓDIGO AQUI

h = model.fit(x=X_train_flatten, y=y_train.T, batch_size=batch_size, epochs=max_epochs, verbose=1)



print("Treinamento finalizado \n")
#CÉLULA KE-LIB-07

#IMPLEMENTE O CÓDIGO AQUI

eval = model.evaluate(X_train_flatten, y_train.T, verbose=0)

print("Erro médio do conjunto de treinamento: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))



eval = model.evaluate(X_test_flatten, y_test.T, verbose=0)

print("Erro médio do conjunto de teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))
X_train, y_train, X_test, y_test, classes = loadDatasets()



X_train = X_train / 255

X_test = X_test / 255



X_train_flatten = X_train.reshape(X_train.shape[0], -1)   # The "-1" makes reshape flatten the remaining dimensions

X_test_flatten  = X_test.reshape(X_test.shape[0], -1)
#CÉLULA KE-LIB-08

layers_dims = [12288, 20, 7, 5, 1] # modelo de 4 camadas densas



tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Limpa o modelo previamente usado, senão irá acumular camadas

K.backend.clear_session()



#Inicializador

init = K.initializers.glorot_uniform()



#IMPLEMENTE O CÓDIGO AQUI

#Criando o otimizador

theOptimizer = K.optimizers.RMSprop()



#Construindo o modelo (topologia)

model = K.models.Sequential()



#Rede Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=20, input_dim=12288, kernel_initializer=init, activation='relu'))

model.add(K.layers.Dense(units=7, kernel_initializer=init, activation='relu'))

model.add(K.layers.Dense(units=5, kernel_initializer=init, activation='relu'))

model.add(K.layers.Dense(units=1, kernel_initializer=init, activation='sigmoid'))



#Compile o modelo - IMPLEMENTE O CÓDIGO AQUI

model.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['acc'])
#CÉLULA KE-LIB-09

#IMPLEMENTE O CÓDIGO AQUI

batch_size = 19

max_epochs = 2500



print("Iniciando treinamento... ")

#IMPLEMENTE O CÓDIGO AQUI

h = model.fit(x=X_train_flatten, y=y_train.T, batch_size=batch_size, epochs=max_epochs, verbose=1)



print("Treinamento finalizado \n")
#CÉLULA KE-LIB-10

#IMPLEMENTE O CÓDIGO AQUI

eval = model.evaluate(X_train_flatten, y_train.T, verbose=0)

print("Erro médio do conjunto de treinamento: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))



eval = model.evaluate(X_test_flatten, y_test.T, verbose=0)

print("Erro médio do conjunto de teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))
X_train, y_train, X_test, y_test, classes = loadDatasets()



image_shape = (64,64,3)



X_train = X_train.reshape(-1, 64, 64, 3)

X_train = X_train / 255



X_test = X_test.reshape(-1, 64, 64, 3)

X_test = X_test / 255
#CÉLULA KE-LIB-11

layers_dims = [12288, 20, 7, 5, 1] # modelo de 4 camadas densas



tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Limpa o modelo previamente usado, senão irá acumular camadas

K.backend.clear_session()



#Inicializador

init = K.initializers.glorot_uniform()



#Criando o otimizador

#IMPLEMENTE O CÓDIGO AQUI

theOptimizer = K.optimizers.Adam()



#Construindo o modelo (topologia)

model = K.models.Sequential()



#Camada convolucional - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=init, activation='relu', input_shape=image_shape))



#Camada convolucional - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=init, activation='relu'))



#MaxPooling - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.MaxPooling2D(pool_size=(2,2)))



#Dropout de 25% na camada anterior - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dropout(0, 25))



#Flatten para entrada da rede densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Flatten())



#Camada Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=100, kernel_initializer=init, activation='relu'))

  

#Dropout de 50% na camada anterior - IMPLEMENTE O CÓDIGO AQUI 

model.add(K.layers.Dropout(0.5))



#Camada densa - IMPLEMENTE O CÓDIGO AQUI     

model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='relu'))

#Camada Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=1, kernel_initializer=init, activation='sigmoid'))



#Compile o modelo - IMPLEMENTE O CÓDIGO AQUI 

model.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['acc'])

model.summary()
#CÉLULA KE-LIB-12



#IMPLEMENTE O CÓDIGO AQUI

batch_size = 19

max_epochs = 50



print("Iniciando treinamento... ")

h = model.fit(x=X_train, y=y_train.T, batch_size=batch_size, epochs=max_epochs, verbose=True)



print("Treinamento finalizado \n")
#CÉLULA KE-LIB-13

#IMPLEMENTE O CÓDIGO AQUI

eval = model.evaluate(X_train, y_train.T, verbose=0)

print("Erro médio do conjunto de treinamento: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))



eval = model.evaluate(X_test, y_test.T, verbose=0)

print("Erro médio do conjunto de teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))
#IMPLEMENTE O CÓDIGO AQUI (1 linha)

plt.plot(range(len(h.history['loss'])), h.history['loss'])
# É possível observar que a rede neural da atividade 2, que utiliza camada convolucional

# tem um melhor resultado que a rede da atividade 1. A atividade 2 tem uma perda menor e uma acurácia maior que a atividade 1. 

# Isso deve porque camadas convolucionais são redes bem adaptadas para processar imagens.
X_train, y_train, X_test, y_test, classes = loadDatasets()



image_shape = (64,64,3)



X_train = X_train.reshape(-1, 64, 64, 3)

X_train = X_train / 255



X_test = X_test.reshape(-1, 64, 64, 3)

X_test = X_test / 255
#CÉLULA KE-LIB-11

layers_dims = [12288, 20, 7, 5, 1] # modelo de 4 camadas densas



tf.logging.set_verbosity(tf.logging.ERROR) #desliga os warnings do tensorflow



#Limpa o modelo previamente usado, senão irá acumular camadas

K.backend.clear_session()



#Inicializador

init = K.initializers.glorot_uniform()



#Criando o otimizador

#IMPLEMENTE O CÓDIGO AQUI

theOptimizer = K.optimizers.Adadelta()



#Construindo o modelo (topologia)

model = K.models.Sequential()



#Camada convolucional - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(1,1), padding='same', activation='relu', input_shape=(64, 64,3)))



#Camada convolucional - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(1,1), padding='same', activation='relu'))



#MaxPooling - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.MaxPooling2D(pool_size=(2,2)))



#Dropout de 25% na camada anterior - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dropout(0, 25))



#Flatten para entrada da rede densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Flatten())



#Camada Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=100, kernel_initializer=init, activation='relu'))

  

#Dropout de 50% na camada anterior - IMPLEMENTE O CÓDIGO AQUI 

#model.add(K.layers.Dropout(0.5))



#Camada densa - IMPLEMENTE O CÓDIGO AQUI     

model.add(K.layers.Dense(units=10, kernel_initializer=init, activation='relu'))

#Camada Densa - IMPLEMENTE O CÓDIGO AQUI

model.add(K.layers.Dense(units=1, kernel_initializer=init, activation='sigmoid'))



#Compile o modelo - IMPLEMENTE O CÓDIGO AQUI 

model.compile(loss='binary_crossentropy', optimizer=theOptimizer, metrics=['acc'])

model.summary()
#CÉLULA KE-LIB-12



#IMPLEMENTE O CÓDIGO AQUI

batch_size = 19

max_epochs = 50



print("Iniciando treinamento... ")

h = model.fit(x=X_train, y=y_train.T, batch_size=batch_size, epochs=max_epochs, verbose=True)



print("Treinamento finalizado \n")
#CÉLULA KE-LIB-13

#IMPLEMENTE O CÓDIGO AQUI

eval = model.evaluate(X_train, y_train.T, verbose=0)

print("Erro médio do conjunto de treinamento: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))



eval = model.evaluate(X_test, y_test.T, verbose=0)

print("Erro médio do conjunto de teste: Perda {0:.4f}, acuracia {1:.4f}".format(eval[0], eval[1]*100))
#IMPLEMENTE O CÓDIGO AQUI (1 linha)

plt.plot(range(len(h.history['loss'])), h.history['loss'])
# Removendo o dropout de 50% foi possível notar no teste que a perda diminuiu e a acurácia aumentou.

# O dropout é uma tecnica para evitar overfiting nas redes, no grafico de perda é possível notar picos,

# isso não aconteceu quando foi utilizado o dropout no experimento anterior.