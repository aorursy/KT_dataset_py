import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
train = train.iloc[:,0:31]

test = test.iloc[:,0:30]
from sklearn.preprocessing import LabelEncoder 

# transforma os dados da coluna diagnosis em dados numericos

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])

train.head()
# x = atributos

# y = classes

x = train.drop('diagnosis', axis=1)

y = train['diagnosis']
x.head(5)
print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



# divide o dataset de trainamento em treinamento e teste, separando 25% em teste

# x = atributos e y = classes

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Modelo de Rede Neural com Múltiplas Camadas

# Versão do professor

from keras.models import Sequential

from keras.layers import Dense
classify = Sequential()

# primeira camada oculta

# units 30 + 1 / 2 ~= 15

classify.add(Dense(units=15, activation="relu", input_dim=30))

classify.add(Dense(units=1, activation="sigmoid"))



classify.compile(optimizer="adam", loss="binary_crossentropy", metrics = ['binary_accuracy'])
# Treinando os dados

# batch_size é o numero de entradas que se processa antes de atualizar os pesos

# epochs é quantas vezes a rede passa por/processa todos os dados

classify.fit(x_train, y_train, batch_size=10, epochs=50)
y_pred = classify.predict(x_test)

y_pred = y_pred > 0.5

y_pred
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

print("Precisão: ", acc)
# Trabalho que eu fiz

import keras

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(activation="relu", input_dim=30, units=15, kernel_initializer="uniform"))

model.add(Dense(activation="relu", units=15, kernel_initializer="uniform"))

model.add(Dense(activation="relu", units=15, kernel_initializer="uniform"))

model.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 10, nb_epoch = 100, validation_data=(x_test, y_test))



score = model.evaluate(x_test, y_test, verbose=0)

print('Erro:', score[0])

print('Precisão:', score[1]*100, "%" )