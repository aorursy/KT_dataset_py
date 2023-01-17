import numpy as np

import pandas as pd

import csv

# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5
data = pd.read_csv(r'../input/data32.csv', header=0)

# file with teacher´s performance evaluation_city os Sao Paulo_basic education level
data.head(3)
data=data.drop(['TX_RESP_Q010.1'], axis=1)
data.tail(3)
data.shape
y = data.loc[:,'TT']
y.shape
data=data.drop(['CO_PROFESSOR'], axis=1)
data.head(1)
data=data.drop(['TT'], axis=1)
data.head(1)
data.shape
X=data.to_numpy()

# https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array

X
y=y.to_numpy()
#Normalização dos dados de entrada -- variável explicativa

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
# Separação entre conjuntos de treino e teste

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
#Definição da ANN e suas camadas

import keras

from keras.models import Sequential

from keras.layers import Dense

# Rede Neural -- 13 entradas e 01 saída, com 2 camadas intermediárias (hidden)

model = Sequential()

model.add(Dense(10, input_dim=13, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
treino = model.fit(X_train, y_train, epochs=100, batch_size=64)
y_pred = model.predict(X_test)

# Conversão das predições

pred = list()

for i in range(len(y_pred)):

    pred.append(np.argmax(y_pred[i]))

# Conversão da parte one hot encoded test label para resultado

test = list()

for i in range(len(y_test)):

    test.append(np.argmax(y_test[i]))
from sklearn.metrics import accuracy_score

a = accuracy_score(pred,test)

print('Accuracy is:', a*100)
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=500, batch_size=64)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('ANN - modelo Acurácia')

plt.ylabel('Acurácia')

plt.xlabel('Epoch')

plt.legend(['Treino', 'Teste'], loc='upper left')

plt.show()
plt.plot(history.history['loss']) 

plt.plot(history.history['val_loss']) 

plt.title('Função de Perda') 

plt.ylabel('Perda') 

plt.xlabel('Epoch') 

plt.legend(['Treino', 'Teste'], loc='upper left') 

plt.show()
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(10,6),random_state=1,max_iter=150)

print(model)

model.fit(X_train,y_train)

pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import metrics

print ('\n Teste usando MLP')

print ("Erro médio absoluto:", metrics.mean_absolute_error(y_test,pred))

print ("Erro médio quadrático:", metrics.mean_squared_error(y_test,pred))

print ("Erro médio raiz quadrada:", np.sqrt(metrics.mean_squared_error(y_test,pred)))
confusion_matrix(y_test,pred)
import pandas as pd

data32 = pd.read_csv("../input/data32.csv")