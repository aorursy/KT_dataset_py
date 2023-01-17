!pip install apyori
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from apyori import apriori



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#El objetivo es buscar reglas de asociación sobre el set de datos de los pasajeros del titanic

#Se comienza cargando los archivos 

#Ocupé el set test dado que el csv que tiene los sobrevivientes tiene las id para ese set



titanic_train = pd.read_csv('../input/test.csv')

pasajero_sobrevive = pd.read_csv('../input/gender_submission.csv')

titanic_train.head()
pasajero_sobrevive.head()
#Se realiza un merge en "PassengerId" para que la tabla tenga si el pasajero sobrevivió

titanic_pasajeros = pd.merge(titanic_train, pasajero_sobrevive, on='PassengerId', how='inner')
titanic_pasajeros.head()
#Reemplazo el valor de si sobrevive por un Sí o No, para evitar la confusión en caso de edad igual a 1.

#Basado en la descripción del dataset para Survival se tiene: 0 = No, 1 = Yes.

#Y el de la clase a la que pertenece el pasajero se reemplaza por lo descrito en el dataset

# 1 = 1st, 2 = 2nd, 3 = 3rd

#Para embarked, C = Cherbourg, Q = Queenstown, S = Southampton



titanic_pasajeros = titanic_pasajeros.replace({'Survived': {0: 'No', 1: 'Sí'}, 

                                               'Pclass': {1: '1st', 2: '2nd', 3: '3rd'},

                                              'Embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}})
titanic_pasajeros.head()
#Se eliminan las columnas que tienen un valor único a lo largo de las filas

# y otras como número de hermanos y padres, para simplificar la lectura de la salida del algoritmo apriori



titanic_reduced = titanic_pasajeros.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'])
titanic_reduced.head()
titanic_reduced.shape
rows, columns = titanic_reduced.shape
records = []

for i in range (0, rows):

    records.append([str(titanic_reduced.values[i,j]) for j in range(0, columns)])
#Para las reglas de asociación se utilizan distintos números a modo de prueba.

association_rules = apriori(records, min_suppport = 0.2, min_confidence = 0.3, min_lift = 1, min_length = 2)

association_results = list(association_rules)
print(len(association_results))
print (association_results[0])

for item in association_results:

    print(item[0], '-->', item[2][0][1], 'support: ', item[1], 'confidence: ', item[2][0][2], 

         'lift: ', item[2][0][3])