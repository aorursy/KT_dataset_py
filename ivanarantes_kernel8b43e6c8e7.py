# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import seaborn as sns

import math







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.




data = pd.read_csv("../input/mushrooms.csv")

data.head(10)



data.info()

data['stalk-root'].value_counts()

#Dropando a coluna stalk-root pois valor corresponde a mais de 30%

data['stalk-shape'].value_counts()
Y = pd.get_dummies(data.iloc[:,0],  drop_first=False)

X = pd.DataFrame()

for each in data.iloc[:,1:].columns:

    dummies = pd.get_dummies(data[each], prefix=each, drop_first=False)

    X = pd.concat([X, dummies], axis=1)
X.head()
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score

from keras import backend as K

from keras.layers import BatchNormalization



def create_model():

    model = Sequential()

    model.add(Dense(20, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    return model
split_size = 0.20

epochs = 100

batch_size = 50



model = create_model()

#Otimizador SGD

sgd = SGD(lr=0.01, momentum=0.7, decay=0, nesterov=False)

model.compile(loss='binary_crossentropy' , optimizer=sgd, metrics=['accuracy'])

history = model.fit(X.values, Y.values, validation_split=split_size, epochs=epochs, batch_size=batch_size, verbose=0)
def listarDados(history):

    # listando dados

    print(history.history.keys())

    # summarize history for accuracy

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['treino', 'validacao'], loc='upper left')

    plt.show()

    # listando loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['treino', 'validacao'], loc='upper left')

    plt.show()
listarDados(history)

print("Acurácia treino: %.2f%% / Acurácia validação: %.2f%%" % 

      (100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))
model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

historyAdam = model.fit(X.values, Y.values, validation_split=split_size, epochs=epochs, batch_size=batch_size, verbose=0)

listarDados(historyAdam)
print("Acurácia treino: %.2f%% / Acurácia validação: %.2f%%" % 

      (100*historyAdam.history['acc'][-1], 100*historyAdam.history['val_acc'][-1]))