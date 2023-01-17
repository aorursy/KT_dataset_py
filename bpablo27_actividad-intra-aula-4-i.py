# Import de bibliotecas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# Carga de datos

df = pd.read_csv('../input/pulsar_stars.csv')
# Análisis de dimensiones del dataset

print('Shape: ', df.shape)

print('Columns: ', df.columns)

df.head
# Matriz de características

x = df.iloc[:, 0:8]

print('Shape de matriz de caracteristicas: ', x.shape)

x
# Vector target

y = df.iloc[:, -1]

print('Target Shape: ' , y.shape)

y

# Paso 2. Importar biblioteca para clasificación mediante KNN



from sklearn.neighbors import KNeighborsClassifier 
# Selección de la cantidad de vecinos 'n'

knn = KNeighborsClassifier(n_neighbors=5)
# Entrenamiento del modelos con las matrices de data y target

knn.fit(x,y)
# Vector de prueba

X_new = [[99, 40, 1.5, 5,3,19,7,65]]
# Resultado de clasificar este nuevo vector.

y_pred = knn.predict(X_new) 

y_pred