# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# @autor: mario ruiz

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Importing dataset
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
# Visualizing first 5 data
print(iris.head())
# Drop column ID
iris = iris.drop('Id', axis=1)
# View features
print('Dataset Information:')
print(iris.info())
print('Describe dataset')
print(iris.describe())
print('Species Distribution:')
print(iris.groupby('Species').size())
import matplotlib as plt
# Visualising Data 
fig = iris [iris.Species == 'Iris-setosa'].plot(kind='scatter',
           x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')
iris [iris.Species == 'Iris-versicolor'].plot(kind='scatter',
           x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor',ax=fig)
iris [iris.Species == 'Iris-virginica'].plot(kind='scatter',
           x='SepalLengthCm', y='SepalWidthCm', color='red', label='Virginica', ax=fig)

fig.set_xlabel('Sepalo - Longitud')
fig.set_ylabel('Sepalo - Ancho')
fig.set_title('Sepalo - Longitud vs Ancho')




fig = iris [iris.Species == 'Iris-setosa'].plot(kind='scatter',
           x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')
iris [iris.Species == 'Iris-versicolor'].plot(kind='scatter',
           x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor',ax=fig)
iris [iris.Species == 'Iris-virginica'].plot(kind='scatter',
           x='PetalLengthCm', y='PetalWidthCm', color='red', label='Virginica', ax=fig)

fig.set_xlabel('Petalo - Longitud')
fig.set_ylabel('Petalo - Ancho')
fig.set_title('Petalo - Longitud vs Ancho')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#Separo todos los datos con las características y las etiquetas o resultados
X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Regresión Logística: {}'.format(algoritmo.score(X_train, y_train)))
#Modelo de Máquinas de Vectores de Soporte
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Máquinas de Vectores de Soporte: {}'.format(algoritmo.score(X_train, y_train)))

#Modelo de Vecinos más Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Vecinos más Cercanos: {}'.format(algoritmo.score(X_train, y_train)))
#Modelo de Árboles de Decisión Clasificación
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Árboles de Decisión Clasificación: {}'.format(algoritmo.score(X_train, y_train)))
#Separo todos los datos con las características y las etiquetas o resultados
sepalo = iris[['SepalLengthCm','SepalWidthCm','Species']]
X_sepalo = np.array(sepalo.drop(['Species'], 1))
y_sepalo = np.array(sepalo['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sepalo, y_sepalo, test_size=0.2)
print('Son {} datos sépalo para entrenamiento y {} datos sépalo para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Regresión Logística - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
#Modelo de Vecinos más Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Vecinos más Cercanos - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
#Modelo de Árboles de Decisión Clasificación
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Árboles de Decisión Clasificación - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
petalo = iris[['PetalLengthCm','PetalWidthCm','Species']]
X_petalo = np.array(petalo.drop(['Species'], 1))
y_petalo = np.array(petalo['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_petalo, y_petalo, test_size=0.2)
print('Son {} datos pétalo para entrenamiento y {} datos pétalo para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Regresión Logística - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de Máquinas de Vectores de Soporte
algoritmo = SVC()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Máquinas de Vectores de Soporte - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de Vecinos más Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Vecinos más Cercanos - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de Árboles de Decisión Clasificación
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Árboles de Decisión Clasificación - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))