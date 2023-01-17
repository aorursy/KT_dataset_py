

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



import os

print(os.listdir("../input"))
path = "../input/"

df_train = pd.read_csv(path+"train.csv")

df_test = pd.read_csv(path+"test.csv")

df_train.head()
df_test.head()
df_train.shape
df_test.shape
df_train.info()
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()
df_train.describe()
df_test.describe()
df_train['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

df_test['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
df_train['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace=True)

df_test['Embarked'].replace(['Q', 'S', 'C'], [0, 1, 2], inplace=True)
df_train['Age'].mean()
df_test['Age'].mean()
promedio = 30

df_train['Age'].replace(np.nan, promedio)

df_test['Age'].replace(np.nan, promedio)
bins = [0, 8, 15, 18, 25, 40, 68, 100]

names = ['1', '2', '3', '4', '5', '6', '7']

df_train['Age'] = pd.cut(df_train['Age'], bins, labels=names)

df_test['Age'] = pd.cut(df_test['Age'], bins, labels=names)
df_train.drop(['Cabin'], axis=1, inplace=True)

df_test.drop(['Cabin'], axis=1, inplace=True)
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

df_test = df_test.drop(['Name', 'Ticket'], axis=1)
df_train.dropna(axis=0, how='any', inplace=True)

df_test.dropna(axis=0, how='any', inplace=True)
pd.isnull(df_train).sum()
pd.isnull(df_test).sum()
df_test.head()
df_train.head()
#Separo la columna con la información de los sobrevivientes

X = np.array(df_train.drop(['Survived'], 1))

y = np.array(df_train['Survived'])
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
##Regresión logística

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

print('Precisión Regresión Logística:')

print(logreg.score(X_train, y_train))
##Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

print('Precisión Soporte de Vectores:')

print(svc.score(X_train, y_train))
##K neighbors

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

print('Precisión Vecinos más Cercanos:')

print(knn.score(X_train, y_train))
ids = df_test['PassengerId']
###Regresión logística

prediccion_logreg = logreg.predict(df_test.drop('PassengerId', axis=1))

out_logreg = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_logreg })

print('Predicción Regresión Logística:')

print(out_logreg.head())
##Support Vector Machines

prediccion_svc = svc.predict(df_test.drop('PassengerId', axis=1))

out_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_svc })

print('Predicción Soporte de Vectores:')

print(out_svc.head())
##K neighbors

prediccion_knn = knn.predict(df_test.drop('PassengerId', axis=1))

out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_knn })

print('Predicción Vecinos más Cercanos:')

print(out_knn.head())
filename = 'TitanicPredictions1.csv'



out_svc.to_csv(filename,index=False)



print('Saved file: ' + filename)