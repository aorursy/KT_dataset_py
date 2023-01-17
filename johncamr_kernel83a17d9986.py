# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import keras

from keras import regularizers

from keras.layers import Dropout

# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist

import os









print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
filedttrain = "../input/fashion-mnist_train.csv"

filedttest  = "../input/fashion-mnist_test.csv"

dftr = pd.read_csv(filedttrain)

dfte = pd.read_csv(filedttest)
print(dftr.info())

print(dfte.info())
print(dftr.head(3))

print(dftr.tail(3))
print(dfte.head(3))

print(dfte.tail(3))
def interpretar(dataframe):

    # Select all columns but the first

    caract = dataframe.values[:, 1:]/255

    # The first column is the label. Conveniently called 'label'

    etiqueta = dataframe['label'].values

    return caract, etiqueta 
car_entr,etiq_entr =interpretar(dftr)

car_pru,etiq_pru =interpretar(dfte)

m_train = car_entr.shape[0]

m_test = car_pru.shape[0]

print("X_train shape: " + str(car_entr.shape))

print("y_train shape: " + str(etiq_entr.shape))

print("X_test shape: " + str(car_pru.shape))

print("y_test shape: " + str(etiq_pru.shape))

print ("# de registros de entrenamiento: m_train = " + str(m_train))

print ("# de registros de prueba: m_test = " + str(m_test))
PruebaIndice = 49999

plt.figure()

_ = plt.imshow(np.reshape(car_entr[PruebaIndice, :], (28,28)), 'gray')
PruebaIndice = 10

plt.figure()

_ = plt.imshow(np.reshape(car_pru[PruebaIndice, :], (28,28)), 'gray')
np.random.seed(0);

PruebaIndice = list(np.random.randint(m_train,size=9))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(car_entr[PruebaIndice[i]].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Index {} Class {}".format(PruebaIndice[i], etiq_entr[PruebaIndice[i]]))

    plt.tight_layout()
etiq_entr.shape
etiq_entr[PruebaIndice]
etiq_entr = tf.keras.utils.to_categorical(etiq_entr)

etiq_pru = tf.keras.utils.to_categorical(etiq_pru)
etiq_entr[PruebaIndice]
var_iteracion=2

var_size=128

var_iteracionV2=15

var_sizeV2=128
##VERSION 1

modelV1 = tf.keras.Sequential()

modelV1.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(784,)))

modelV1.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))

modelV1.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
##VERSION 2

modelV2 = tf.keras.Sequential()

modelV2.add(tf.keras.layers.Dense(60, activation=tf.nn.relu, input_shape=(784,)))

modelV2.add(tf.keras.layers.Dense(50, activation=tf.nn.sigmoid))

modelV2.add(tf.keras.layers.Dense(40, activation=tf.nn.tanh))

#modelV2.add(tf.keras.layers.Dense(20, activation=tf.nn.tanh,kernel_regularizer=regularizers.l2(0.01),

#                activity_regularizer=regularizers.l1(0.01)))

modelV2.add(tf.keras.layers.Dense(30,Dropout(0.4)))

modelV2.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
P_lossV1='categorical_crossentropy'
P_optimizerV1 = tf.train.RMSPropOptimizer(learning_rate=0.005)
P_optimizerV2 = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
modelV1.compile(loss=P_lossV1,

              optimizer=P_optimizerV1,

              metrics=['accuracy'])
modelV2.compile(loss=P_lossV1,

              optimizer=P_optimizerV2,

              metrics=['accuracy'])
modelV1.summary()
modelV2.summary()
Hist1=modelV1.fit(car_entr,etiq_entr, epochs=var_iteracion, batch_size=var_size)
modelV2.fit(car_entr,etiq_entr, epochs=var_iteracionV2, batch_size=var_sizeV2)
# evaluamos el modelo

scores = modelV1.evaluate(car_pru, etiq_pru)

 

print("\n%s: %.2f%%" % (modelV1.metrics_names[1], scores[1]*100))

print (modelV1.predict(car_pru).round())
scores = modelV2.evaluate(car_pru, etiq_pru)

 

print("\n%s: %.2f%%" % (modelV2.metrics_names[1], scores[1]*100))

print (modelV2.predict(car_pru).round())