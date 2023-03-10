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
#Separo todos los datos con las caracter??sticas y las etiquetas o resultados
X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresi??n Log??stica
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisi??n Regresi??n Log??stica: {}'.format(algoritmo.score(X_train, y_train)))
#Modelo de M??quinas de Vectores de Soporte
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisi??n M??quinas de Vectores de Soporte: {}'.format(algoritmo.score(X_train, y_train)))

#Modelo de Vecinos m??s Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisi??n Vecinos m??s Cercanos: {}'.format(algoritmo.score(X_train, y_train)))
#Modelo de ??rboles de Decisi??n Clasificaci??n
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisi??n ??rboles de Decisi??n Clasificaci??n: {}'.format(algoritmo.score(X_train, y_train)))
#Separo todos los datos con las caracter??sticas y las etiquetas o resultados
sepalo = iris[['SepalLengthCm','SepalWidthCm','Species']]
X_sepalo = np.array(sepalo.drop(['Species'], 1))
y_sepalo = np.array(sepalo['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sepalo, y_sepalo, test_size=0.2)
print('Son {} datos s??palo para entrenamiento y {} datos s??palo para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresi??n Log??stica
algoritmo = LogisticRegression()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisi??n Regresi??n Log??stica - S??palo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
#Modelo de Vecinos m??s Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisi??n Vecinos m??s Cercanos - S??palo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
#Modelo de ??rboles de Decisi??n Clasificaci??n
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisi??n ??rboles de Decisi??n Clasificaci??n - S??palo: {}'.format(algoritmo.score(X_train_s, y_train_s)))
petalo = iris[['PetalLengthCm','PetalWidthCm','Species']]
X_petalo = np.array(petalo.drop(['Species'], 1))
y_petalo = np.array(petalo['Species'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_petalo, y_petalo, test_size=0.2)
print('Son {} datos p??talo para entrenamiento y {} datos p??talo para prueba'.format(X_train.shape[0], X_test.shape[0]))
#Modelo de Regresi??n Log??stica
algoritmo = LogisticRegression()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisi??n Regresi??n Log??stica - P??talo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de M??quinas de Vectores de Soporte
algoritmo = SVC()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisi??n M??quinas de Vectores de Soporte - P??talo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de Vecinos m??s Cercanos
algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisi??n Vecinos m??s Cercanos - P??talo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
#Modelo de ??rboles de Decisi??n Clasificaci??n
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisi??n ??rboles de Decisi??n Clasificaci??n - P??talo: {}'.format(algoritmo.score(X_train_p, y_train_p)))