# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importando as livrarias

import pandas as pd                

import numpy as np                 

import seaborn as sns              

import matplotlib.pyplot as plt    

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

%matplotlib inline                 
# Lendo o CSV

data = pd.read_csv('/kaggle/input/titanic/train.csv')

backup = data.copy()

# Quantidade de linhas e colunas do dataset

print(data.shape)
#Lendo csv teste

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(test.shape)
#Visão rápida dos dados

data.head()
#Verificar Nulls

data.isnull().sum()
#Estadísticas básicas del conjunto de datos

data.describe()
#Correlação dos dados

data.corr()
#Eliminamos las columnas que resolvimos no incluir en el análisis

data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
# Asignamos NNN a los registros vacíos de la columna Embarked

data['Embarked'].fillna("NNN", inplace=True)

# Ignoramos los registros NNN de la columna Embarked

data = data.loc[data['Embarked'] != "NNN", :]
#Removendo a feature Cabin

data.drop(['Cabin'], axis=1, inplace=True)
#Verificando valores nulls

data.isnull().sum()
#Intentando saber la edad de los faltantes, decidimos guiarnos por los títulos de cada uno

# Utilizamos el método EXTRACT para hacer una Expresión regular 

# Creamos una columna (feature) llamada "Title" para guardar los títulos de los pasajeros

# La idea es verificar si la media de las edades varía con los títulos, y usar la média para llenar los valores ausentes



data['Title'] = ''

for i in data:

    data['Title'] = backup['Name'].str.extract('([A-Za-z]+)\.', expand=False)
# Creamos un diccionario usando los titulos como llave y las medias de edad como valor

age_means = data.groupby('Title')['Age'].mean().to_dict()

age_means
# Encontramos, usando loc, las lineas que no tienen edad definida e llenamos esas lineas

# con los valores médios.



no_age = data.loc[np.isnan(data['Age'])].index

data.loc[no_age, 'Age'] = data['Title'].loc[no_age].map(age_means)
# Ahora los datos están completos



data.isnull().sum()
#Aplicação de Dummies

data = pd.concat([data,pd.get_dummies(data['Sex'],prefix='Sex')], axis=1)

data.drop(['Sex'],axis=1, inplace=True)



data = pd.concat([data,pd.get_dummies(data['Embarked'],prefix='Embarked')], axis=1)

data.drop(['Embarked'],axis=1, inplace=True)



data = pd.concat([data,pd.get_dummies(data['Pclass'],prefix='Pclass')], axis=1)

data.drop(['Pclass'],axis=1, inplace=True)
#Removendo a feature Title

data.drop(['Title'],axis = 1, inplace = True)
#Revisão dos dados

data.head(10)
#Normalizando as features

data['Age'] = (data['Age'] - data['Age'].min()) / (data['Age'].max()- data['Age'].min())

data['Fare'] = (data['Fare'] - data['Fare'].min()) / (data['Fare'].max() - data['Fare'].min())

data['SibSp'] = (data['SibSp'] - data['SibSp'].min()) / (data['SibSp'].max() - data['SibSp'].min())

data.drop(['Parch'], axis=1, inplace=True)

#data['Parch'] = (data['Parch'] - data['Parch'].min()) / (data['Parch'].max() - data['Parch'].min())
data.head()
#Separação de X e Y

X = data.loc[: , data.columns != 'Survived']

y = data['Survived']









#param ={"eta": 0.2,

#    "max_depth": 4,

#    "objective": "binary:logistic",

#    "silent": 1,

#    "base_score": np.mean(train_y),

#    'n_estimators': 500,

#    "eval_metric": "logloss",

#       'class_weight':'balanced'}    
#definir os parámetros da randomforest

parameters = {'bootstrap': True,

              'min_samples_leaf': 3,

              'n_estimators': 500, 

              'min_samples_split': 10,

              'max_features': 'sqrt',

              'max_depth': 6,

              'max_leaf_nodes': None,

             'class_weight':'balanced'}
#Definir a randomforest

from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(**parameters)
#Treinando o Modelo

RF_model.fit(X, y)
#Trabajando con datos de teste

test_data = pd.DataFrame()

test_data = test

test_data.head()
#Preparando datos para la presentación

final_output = pd.DataFrame()

final_output = pd.DataFrame({'PassengerId': test_data['PassengerId']})

final_output.head()
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.head()
# Fill NaN value with most common Port

test_data['Fare'] = test_data['Fare'].fillna((test_data['Fare'].mean()))



#Verificar Nulls

test_data.isnull().sum()
test_data['Title'] = ''

for i in test_data:

    test_data['Title'] = backup['Name'].str.extract('([A-Za-z]+)\.', expand=False)
age_means = test_data.groupby('Title')['Age'].mean().to_dict()

age_means


no_age = test_data.loc[np.isnan(test_data['Age'])].index

test_data.loc[no_age, 'Age'] = test_data['Title'].loc[no_age].map(age_means)
test_data.isnull().sum()
#Aplicação de Dummies

test_data = pd.concat([test_data,pd.get_dummies(test_data['Sex'],prefix='Sex')], axis=1)

test_data.drop(['Sex'],axis=1, inplace=True)



test_data = pd.concat([test_data,pd.get_dummies(test_data['Embarked'],prefix='Embarked')], axis=1)

test_data.drop(['Embarked'],axis=1, inplace=True)



test_data = pd.concat([test_data,pd.get_dummies(test_data['Pclass'],prefix='Pclass')], axis=1)

test_data.drop(['Pclass'],axis=1, inplace=True)
test_data.drop(['Title'],axis = 1, inplace = True)
#Normalizando as features

test_data['Age'] = (test_data['Age'] - test_data['Age'].min()) / (test_data['Age'].max()- test_data['Age'].min())

test_data['Fare'] = (test_data['Fare'] - test_data['Fare'].min()) / (test_data['Fare'].max() - test_data['Fare'].min())

test_data['SibSp'] = (test_data['SibSp'] - test_data['SibSp'].min()) / (test_data['SibSp'].max() - test_data['SibSp'].min())

test_data.drop(['Parch'], axis=1, inplace=True)

#test_data['Parch'] = (test_data['Parch'] - test_data['Parch'].min()) / (test_data['Parch'].max() - test_data['Parch'].min())






test_data.head()
test_X = test_data[test_data.columns].values


test_Y = RF_model.predict(test_X)
final_output['Survived'] = test_Y

final_output.head(20)
# Criação de csv de saída.

final_output.to_csv('RandomForestModel_out_03.csv',sep=',',index=False)