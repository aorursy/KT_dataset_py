# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.isnull().sum().sort_values(ascending = False)
test_df.isnull().sum().sort_values(ascending = False)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)

test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
#Se elimina Cabin porque le faltan muchos datos

train_df.drop(['Cabin'], axis = 1, inplace=True)

test_df.drop(['Cabin'], axis = 1, inplace=True)
#Completamos la edad con la mediana

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
#Verificación de datos

print(pd.isnull(train_df).sum())

print(pd.isnull(test_df).sum())

print(train_df.shape)

print(test_df.shape)

print(test_df.head())

print(train_df.head())
#Se elimina las filas con los datos perdidos

train_df.dropna(axis=0, how='any', inplace=True)

test_df.dropna(axis=0, how='any', inplace=True)
print(test_df['Embarked'])
#Creo varios grupos de acuerdo a bandas de las edades

#Bandas: 0-8, 9-15, 16-18, 19-25, 26-40, 41-60, 61-100

bins = [0, 8, 15, 18, 25, 40, 60, 100]

names = ['little_boys', 'big_boys', 'teen', 'young_adults', 'adults', 'advanced_adults', 'elderly']

train_df['Age'] = pd.cut(train_df['Age'], bins, labels = names)

test_df['Age'] = pd.cut(test_df['Age'], bins, labels = names)
train_df['Age'].head(10)
#One hot enconding con las columnas

train_df = pd.concat([train_df, pd.get_dummies(train_df['Age'])], axis=1)
#One hot enconding con las columnas

test_df = pd.concat([test_df, pd.get_dummies(test_df['Age'])], axis=1)
train_df.teen.head(30)
train_df.info()
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
train_df.Name.head(10)
#los títulos siempre se encuentran entre una coma y un punto, buscamos los strings entre esos dos y lo guardamos en 

#una nueva columna llamada Title.

train_df['Title'] = train_df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# inspect the amount of people for each title

train_df['Title'].value_counts()
# Lo mismo para las de test

test_df['Title'] = test_df.Name.map( lambda x: x.split(',')[1].split( '.' )[0].strip())



# inspect the amount of people for each title

test_df['Title'].value_counts()
train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')

train_df['Title'] = train_df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

train_df.Title.loc[ (train_df.Title !=  'Master') & (train_df.Title !=  'Mr') & (train_df.Title !=  'Miss') 

             & (train_df.Title !=  'Mrs')] = 'Others'
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')

test_df['Title'] = test_df['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

test_df.Title.loc[ (test_df.Title !=  'Master') & (test_df.Title !=  'Mr') & (test_df.Title !=  'Miss') 

             & (test_df.Title !=  'Mrs')] = 'Others'
train_df['Title'].value_counts()
test_df['Title'].value_counts()
# inspect the correlation between Title and Survived

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#One hot enconding con las columnas

train_df = pd.concat([train_df, pd.get_dummies(train_df['Title'])], axis=1).drop(labels=['Name'], axis=1)
#One hot enconding con las columnas

test_df = pd.concat([test_df, pd.get_dummies(test_df['Title'])], axis=1).drop(labels=['Name'], axis=1)
train_df.info()
test_df.info()
#Se elimina la columna de "Title"

train_df.drop(['Title'], axis = 1, inplace=True)

test_df.drop(['Title'], axis = 1, inplace=True)
train_df['FamilySize'].head(10)
train_df['Sex'].head(10)
# Se mapean los valores de sex Male=1 y Female=0

train_df.Sex = train_df.Sex.map({'male':1, 'female':0})

test_df.Sex = test_df.Sex.map({'male':1, 'female':0})
train_df['Sex'].head(10)
train_df.head(10)
#Se elimina la columna de "Age"

train_df.drop(['Age'], axis = 1, inplace=True)

test_df.drop(['Age'], axis = 1, inplace=True)
#Cambio las letras de embarqued a números

train_df['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)

test_df['Embarked'].replace(['Q','S', 'C'],[0,1,2],inplace=True)
train_df.head(10)
#Se elimina la columna de "Ticket" porque no la voy a usar

train_df.drop(['Ticket'], axis = 1, inplace=True)

test_df.drop(['Ticket'], axis = 1, inplace=True)
#Se elimina la columna de "PassengerId" del train_df

train_df.drop(['PassengerId'], axis = 1, inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
sns.heatmap(train_df.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 

#data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()
from sklearn.model_selection import train_test_split #para splitear los datos



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import accuracy_score  #para puntaje de presición

from sklearn.model_selection import KFold #para validación cruzada K-fold

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #para confusion matrix

from sklearn.model_selection import GridSearchCV



#Separar la columna con la información que queremos predecir 

#para luego poder comparar el valor real con los valores de la predicción

all_features = train_df.drop("Survived",axis=1)

targeted_feature = train_df["Survived"]



#Separar los datos de "train" en entrenamiento y prueba para probar los algoritmos

X_train,X_test,y_train,y_test = train_test_split(all_features,targeted_feature,test_size=0.2)

model = RandomForestClassifier(criterion='gini', n_estimators=700,

                            min_samples_split=10,min_samples_leaf=1,

                            max_features='auto',oob_score=True,

                            random_state=1,n_jobs=-1)

model.fit(X_train,y_train)

prediction_rm=model.predict(X_test)

print('--------------The Accuracy of the model----------------------------')

print('The accuracy of the Random Forest Classifier is', round(accuracy_score(prediction_rm,y_test)*100,2))
#RandomForest Fine Tuning



# Random Forest Classifier Parameters tunning

model = RandomForestClassifier()

n_estim=range(100,1000,100)

## Search grid for optimal parameters

param_grid = {"n_estimators" :n_estim}

model_rf = GridSearchCV(model,param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)

model_rf.fit(X_train,y_train)

# Best score

print(model_rf.best_score_)

#best estimator

model_rf.best_estimator_



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
#cargo los ids

ids = test_df['PassengerId']
##Regresión logística

prediccion_logreg = logreg.predict(test_df.drop('PassengerId', axis=1))

out_logreg = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_logreg })

print('Predicción Regresión Logística:')

print(out_logreg.head())
##Support Vector Machines

prediccion_svc = svc.predict(test_df.drop('PassengerId', axis=1))

out_svc = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_svc })

print('Predicción Soporte de Vectores:')

print(out_svc.head())
##K neighbors

prediccion_knn = knn.predict(test_df.drop('PassengerId', axis=1))

out_knn = pd.DataFrame({ 'PassengerId' : ids, 'Survived': prediccion_knn })

print('Predicción Vecinos más Cercanos:')

print(out_knn.head())
out_knn.to_csv('submission.csv', index=False)