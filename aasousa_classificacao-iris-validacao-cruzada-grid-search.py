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
base.head()
base[['sepal_length','sepal_width',	'petal_length',	'petal_width']].boxplot()
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential # camadas sequencialmente
from keras.layers import Dense # camadas profundas fullconnect ou densa
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# 3 tipos em uma coluna
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
# (quantidade de entrada + quantidade de saída) / 2
units = np.round((4 + 3) / 2)
units
def criarRede(optimizer, 
              loss, 
              kernel_initializer, 
              activation, 
              neurons):
  classificador = Sequential();
  classificador.add(Dense(units = neurons, 
                        activation = activation,
                        kernel_initializer = kernel_initializer,
                        input_dim = 4));

  classificador.add(Dense(units = neurons, 
                          kernel_initializer = kernel_initializer,
                          activation = activation));

  # Função softmax retorna probabilidade para cada classe
  classificador.add(Dense(units=neurons, activation = 'softmax'));

  classificador.compile(optimizer = optimizer, 
                      loss = loss,
                      metrics = ['accuracy']);
  return classificador;
classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [5, 50],
             'epochs': [10, 100],
             'optimizer': ['adam', 'sgd'],
             'loss': ['categorical_crossentropy', 'sparse_categorical_crossentropy'],
             'kernel_initializer': ['uniform','random_uniform'],
             'activation': ['relu', 'tanh'],
             'neurons': [4, 8]}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = classificador,
                          param_grid = parametros,
                          cv = 10)
grid_search = grid_search.fit(previsores, classe);
melhores_parametros = grid_search.best_params_
melhores_parametros
melhor_precisao = grid_search.best_score_
melhor_precisao