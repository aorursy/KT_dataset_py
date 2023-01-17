# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import SGD

from keras.utils import np_utils





# Caricamento delle librerie generiche

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
# Creo un validation set da passare come parametro alla rete grande come il 10% dell'intero train usando TRAIN-TEST-SPLIT



T,Val=train_test_split(train, test_size=0.05, train_size=0.95)



# il .value è per far si che siano dei numpy array visto che le KERAS cnn non prendono in input i DATAFRAME

T_y=T.label.values

T_x=T.drop(['label'],axis=1).values



Val_y=Val.label.values

Val_x=Val.drop(['label'],axis=1).values



# Ora per poter mandare questi dati in ingresso alla rete si deve fare un reshape in 28x28
# Ora sono array di matrici 28 x 28



T_x=T_x.reshape(-1,1,28,28)

Val_x=Val_x.reshape(-1,1,28,28)



print (T_x.shape)

print (Val_x.shape)
T_y=np_utils.to_categorical(T_y,10)

Val_y=np_utils.to_categorical(Val_y,10)
# provo con un semplice layer convoluzionale e uno di pooling per iniziare



model=Sequential()



#Convoluzionale

model.add(Convolution2D(32,3,3,input_shape=(1,28,28),activation='relu',border_mode='same'))

# usiamo 32 filtri di dimensione 3x3 , va specificato la dimensione dell'input che nel mio caso sono

# le immagini 28x28x1(che è come se fosse la profondità--> se ci fossero i valori RGB cioè 3 valori per pixel avrei x3),

# come activation function si potrebbe utilizzare la sigmoide ma si preferisce la RELU che è una sigmoide migliorata per le rete neurali



# Visto che il contenuto informativo del bordo nelle nostre immagini non è troppo importante(Spesso sono pixel neri)

# Anche se potremmo adottare una gestione FULL (perchè utilizziamo THEANO) utilizziamo un 'same'



model.add(Convolution2D(32, 3, 3, border_mode='same'))



#Pooling 

# Ora non vanno speicificate le dimensioni dell'input perchè si adatta automaticamente a quelle del layer 

# inserito con l'ultimo add



model.add(MaxPooling2D(pool_size=(2,2),dim_ordering='th'))



model.add(Dropout(0.3))



# Ora è necessario utilizzare un FLATTEN layer per riportare il tutto in maniera vettoriale e mandare in ingresso 

# a una FULL CONNETTED NN , non specifico parametri del FLATTEN perchè dovrebbe settari in automatico



model.add(Flatten())



# Ora inserisco un layer DENSE ovvero il primo della FCNN con 64 nodi e activation sempre RELU



model.add(Dense(128,activation='relu'))



# Utilizzando un layer di pooling appena prima del FLatten si ottiene un errore perchè si sta riducendo di troppo

# la dimensione e quindi flatten ottiene un input con una dimensione = 0 , per ora risolvo il problema ignorando

# il layer di pooling ( altrimenti potrei espandere le dimensioni due volte --> due convulutional2D prima del pooling)



# Ora va inserito l'ultimo dense layer per la classificazione con 10 nodi e l'activation SOFTMAX in modo da avere 

# su ogni nodo la probabilità di ogni classe

model.add(Dropout(0.3))

model.add(Dense(10,activation='softmax'))



# Ora che ho inserito i layer nella rete si deve
# COMPILE :

# - Specificare l'ottimizzatore ----> Stocasthic gradient descend in prima istanza

# - il tipo di loss che sarà categorical_crossentropy perchè un porblema multi classe e per l'appunto utilizziamo quella

# rappresentazione delle label vista prima

# - e la metrica che sarà ACCURACy

model.compile(optimizer='adadelta',loss='categorical_crossentropy', metrics=['accuracy'])
batches=128 # numero di campioni per ogni ottimizzazione

epochs= 28 # Numero dei passaggi 



hist=model.fit(T_x,T_y, batch_size=batches, nb_epoch=epochs, validation_data=(Val_x,Val_y),verbose=0)


# Traccio un grafico del variare della LOSS  e della ACCURACY con le iterazioni

# sfruttando .history() di hist . Ma prima faccio un .evalutate su val_x e val_y



scores = model.evaluate(T_x,T_y, verbose=0)



print (scores)
# tracciamento del grafico della LOSS e VAL-LOSS al passo di iterazione

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('Loss Rate')

plt.ylabel('Loss')

plt.xlabel('Training interations')

plt.show()
# E tracciamento del gradfico degli score



plt.plot(hist.history['val_acc'])

plt.title('Accuracy Rate')

plt.ylabel('Accuracy %')

plt.xlabel('Training iterations')

plt.show()
scores = model.evaluate(Val_x,Val_y, verbose=0)



print (scores)
# Caricamento e predizione sul test

test=test.values

test=test.reshape(-1,1,28,28)

predicition=model.predict_classes(test,verbose=0)
Risultati=pd.DataFrame()

Risultati['ImageId']=list(range(1,len(test)+1))

Risultati['Label']=predicition

Risultati.to_csv('result_CNN.csv',index=False)