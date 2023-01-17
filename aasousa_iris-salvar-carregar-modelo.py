# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv', sep=',')
base.head(10)
base['species'].value_counts()
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
params = {'activation': 'relu',
 'batch_size': 5,
 'epochs': 100,
 'kernel_initializer': 'random_uniform',
 'loss': 'sparse_categorical_crossentropy',
 'neurons': 8,
 'optimizer': 'adam'}
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
# 3 tipos em uma coluna
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
classificador = Sequential();
classificador.add(Dense(units = params['neurons'], 
                    activation = params['activation'],
                    kernel_initializer = params['kernel_initializer'],
                    input_dim = 4));

classificador.add(Dense(units = params['neurons'], 
                      kernel_initializer = params['kernel_initializer'],
                      activation = params['activation']));

# Função softmax retorna probabilidade para cada classe
classificador.add(Dense(units=params['neurons'], activation = 'softmax'));

classificador.compile(optimizer = params['optimizer'], 
                  loss = params['loss'],
                  metrics = ['accuracy']);
classificador.fit(previsores, classe, batch_size = params['batch_size'], epochs = params['epochs'])
classificador_json = classificador.to_json()
classificador_json
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')
from keras.models import model_from_json
arquivo_modelo_treinado = open('classificador_iris.json', 'r');
estrutura_modelo_treinado = arquivo_modelo_treinado.read()
estrutura_modelo_treinado
arquivo_modelo_treinado.close
classificador_modelo_treinado = model_from_json(estrutura_modelo_treinado)
classificador_modelo_treinado.load_weights('classificador_iris.h5')
classificador_modelo_treinado
# Nova entrada de dados para análise
novo = base.sample(1)
novo.head()
previsores = novo.iloc[:, 0:4].values
classe = novo.iloc[:, 4].values
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
previsao = classificador_modelo_treinado.predict(previsores)
previsao
previsao = (previsao > 0.5)
previsao

classificador_modelo_treinado.compile(loss='sparse_categorical_crossentropy', 
                                optimizer='adam',
                                metrics=['accuracy'])

# Realizando a avaliação do modelo, pode ser com qualquer dataset
resultado_modelo_carregado = classificador_modelo_treinado.evaluate(previsores, classe)
resultado_modelo_carregado
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')