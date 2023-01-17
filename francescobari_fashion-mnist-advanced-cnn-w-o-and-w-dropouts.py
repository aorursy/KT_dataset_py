#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:21:26 2020

@author: Francesco Bari

"""



#Import librerie
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



#Dataset

#Ogni immagine è 28x28 pixels (tot=784).
#Ogni pixel ha un valore ad esso associato (tra 0 e 255).
#Più alto è il numero, più il pixel è scuro.
#Ogni set (train & test) è formato da 785 colonne.
#La prima colonna consiste nelle label delle classi (10 in tutto).
#Le altre 784, come detto prima, contengono i valori dei pixel associati ad ogni img



#Parametri 

IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 2018
#Model
NO_EPOCHS = 50
BATCH_SIZE = 128

IS_LOCAL = False

import os

if(IS_LOCAL):
    PATH="../input/fashionmnist/"
else:
    PATH="../input/"
print(os.listdir(PATH))


#I dataset per il train e per il test sono dati in due file separati 
train_file = PATH+"fashion-mnist_train.csv"
test_file  = PATH+"fashion-mnist_test.csv"

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


#Creiamo una mappatura per ogni tipo di classe
#Le 10 classi con le rispettive label sono composte come segue: 
# 
#  0: T-shirt/top
#  1: Trouser
#  2: Pullover
#  3: Dress
#  4: Coat
#  5: Sandal
#  6: Shirt
#  7: Sneaker
#  8: Bag
#  9: Ankle boot
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}



#Data Pre-processing

#Ridefiniamo le colonne. Al posto di 784, le raggruppiamo in (28,28,1).
#Salviamo inoltre le label in un vettore separato
def data_preprocessing(raw):
    out_y = keras.utils.to_categorical(raw.label, NUM_CLASSES)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

#Facciamo questa operazione per entrambi i set
X, y = data_preprocessing(train_data)
X_test, y_test = data_preprocessing(test_data)

#Dividiamo il tranining set in due set:
#80% dati per il train
#20% dati per la validazione
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#Alla fine di questo processo otteremo:
#Fashion MNIST train -  rows: 48000  columns: (28, 28, 1)
#Fashion MNIST valid -  rows: 12000  columns: (28, 28, 1)
#Fashion MNIST test -  rows: 10000  columns: (28, 28, 1)









#Costruzione del Modello

#Usiamo un modello Sequenziale: 
#è definito come una pila lineare di strati (linear stack of layers)
#Lo inizializziamo e poi utilizziamo il metodo "add" per aggiungerci layer.

#Nello specifico aggiungeremo:

#Conv2D: è un 2D Convolutional layer. 
#Parametrizzato come segue:
#filters - il numero di filtri (Kernel) usati con questo layer (32)
#kernel_size - la dimensione del Kernel (3 x 3)
#activation - la funzione di attivazione (relu)
#kernel_initializer - funzione usata per iniziallizare il Kernel
#input_shape - è il formato dell'immagine (28 x 28)

#MaxPooling2D è un Max pooling operation per dati nello spazio a due dimensioni.
#Parametri:
#pool_size - rappresenta il fattore di downsclae in entrambe le direzioni (2 x 2)

#Un secondo Conv2D, con parametri:
#filters - 64
#kernel_size - (3 x 3)
#activaction: relu;

#Un secondo MaxPooling2D, con parametri:
#pool_size - (2 x 2)

#Un terzo Conv2D, con parametri:
#filters - 128
#kernel_size - (3 x 3)
#activaction: relu;


#Flatten. Questo layer appiattisce l'input in una sola dimensione. 
#Senza parametri.

#Dense. Fully-connected Neural Network layer. Parametri:
#units - spazio dimensonale dell'output, ossia il numero di classi (128)
#activation - funzione di attivazione (relu)

#Secondo Dense. Layer finale (fully-connected). Parametri:
#units - 10
#activation - (softmax), viene usata come standard nel final layer.


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, 
                 kernel_size=(3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))


#Compiliamo il modello, specificando i parametri:
#loss - keras.losses.categorical_crossentropy
#optimazer - adam
#metrics - accuracy.

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])



#Inspect del modello
model.summary()

#E relativa stampa
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))








#Addestramento

#Addestriamo il modello con i dati di train, e usiamo il validation set
#(il sub test dei train) per la validazione.

train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))





#Test & Accuracy

#Utilizziamo i dati del dataset test per effettuare le previsioni
#e per valutarne l'accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Circa del 91%



plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(train_model.history['loss'], label='Loss')
plt.plot(train_model.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training - Loss Function')

plt.subplot(2, 2, 2)
plt.plot(train_model.history['acc'], label='Accuracy')
plt.plot(train_model.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Train - Accuracy')








#Nuovo Model 
#Aggiungiamo vari Dropout layers per cercare di evitare overfitting.

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=(IMG_ROWS, IMG_COLS, 1)))
model.add(MaxPooling2D((2, 2)))
#Primo Droput
model.add(Dropout(0.25))
model.add(Conv2D(64, 
                 kernel_size=(3, 3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Secondo Dropout
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
#Terzo Dropout
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#Quarto Dropout
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))



#E lo compiliamo
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


#Vediamo come è composto
model.summary()

#E lo stampiamo
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))







#Addestramento

train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))


#Test & Accuracy

#Utilizziamo i dati del dataset test per effettuare le previsioni
#e per valutarne l'accuracy
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#Circa del 93%


plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(train_model.history['loss'], label='Loss')
plt.plot(train_model.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training - Loss Function')

plt.subplot(2, 2, 2)
plt.plot(train_model.history['acc'], label='Accuracy')
plt.plot(train_model.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.title('Train - Accuracy')