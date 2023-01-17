# imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn.model_selection import cross_val_score

import time



# filepaths

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Carregando a base de treino

train_data = pd.read_csv ("/kaggle/input/adult-pmr3508/train_data.csv",

                         names=[

                         "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

                         "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

                         "Hours per Week", "Country", "Target"],

                         skipinitialspace=True,

                         engine='python',

                         na_values="?",

                         skiprows=1)



# Preenchendo os dados faltantes com a moda

train_data = train_data.fillna (train_data.mode().iloc[0])

train_data.head()
# Dados não numéricos

from sklearn import preprocessing

train_data = train_data.apply (preprocessing.LabelEncoder().fit_transform)



# Atributos utilizados

Xtrain = train_data[["Age", "Education-Num", "Workclass", "Martial Status", "Occupation",

                   "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per Week"]]

Ytrain = train_data.Target # Mantem o formato '<=50K'/'>50K'
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier() # default: gini index



# Utilizando validação cruzada

start = time.time ()

tree_scores = cross_val_score(tree_clf, Xtrain, Ytrain, cv=10)

end = time.time ()

print ("[Tree] Time elapsed:", end-start)

tree_scores
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=10)



# Utilizando validação cruzada

start = time.time ()

forest_scores = cross_val_score(forest_clf, Xtrain, Ytrain, cv=10)

end = time.time ()

print ("[Forest] Time elapsed:", end-start)

forest_scores
from sklearn.ensemble import AdaBoostClassifier

boost_clf = AdaBoostClassifier(n_estimators=100) # stumps



# Utilizando validação cruzada

start = time.time ()

boost_scores = cross_val_score(boost_clf, Xtrain, Ytrain, cv=10)

end = time.time ()

print ("[Boost] Time elapsed:", end-start)

boost_scores
# com 50 stumps

boost_clf2 = AdaBoostClassifier(n_estimators=50) # stumps



# Utilizando validação cruzada

start = time.time ()

boost_scores = cross_val_score(boost_clf2, Xtrain, Ytrain, cv=10)

end = time.time ()

print ("[Boost] Time elapsed:", end-start)

boost_scores