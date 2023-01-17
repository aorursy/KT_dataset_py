# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib as plt

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head(10)
previsores = df.drop('Survived', axis=1)

classe = df['Survived']



previsores.isnull().sum().sort_values(ascending=False)



previsores['Age'].fillna(previsores['Age'].mean(),inplace=True)

previsores.loc[previsores.Embarked.isnull()]

previsores['Embarked'].describe()

previsores['Embarked'] = previsores['Embarked'].fillna('S')

previsores = previsores.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer



colum_transformer = ColumnTransformer([('OneHot', OneHotEncoder(), [1,6])], 

                                       remainder='passthrough')

previsores = colum_transformer.fit_transform(previsores)

scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(

        previsores, classe, test_size=0.25, random_state=0)
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb

classificador =  xgb.XGBClassifier(objective='binary:logistic', random_state=42)

classificador.fit(previsores_treinamento,classe_treinamento)

previsoes = classificador.predict(previsores_teste)

print(accuracy_score(classe_teste, previsoes))

print(confusion_matrix(classe_teste, previsoes))
from sklearn.svm import SVC

classificador_svc = SVC()

classificador_svc.fit(previsores_treinamento, classe_treinamento)

previsoes_svc = classificador_svc.predict(previsores_teste)

print(accuracy_score(classe_teste, previsoes_svc))

print(confusion_matrix(classe_teste, previsoes_svc))
from sklearn.ensemble import RandomForestClassifier

classificador_random = RandomForestClassifier()

classificador_random.fit(previsores_treinamento, classe_treinamento)

previsoes_random = classificador_random.predict(previsores_teste)

print(accuracy_score(classe_teste, previsoes))

print(confusion_matrix(classe_teste, previsoes))