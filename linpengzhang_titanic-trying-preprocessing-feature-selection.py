# Packages

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn import svm

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import math

import timeit

import statistics

# Load the dataset

data = pd.read_csv('../input/TitanicDataset/titanic_data.csv').drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

data.describe()
# Onehot encoding for categorical features

data = pd.concat([data, pd.get_dummies(data.Sex, drop_first=False).astype(int)], axis=1)

data = pd.concat([data, pd.get_dummies(data.Embarked, drop_first=False).astype(int)], axis=1).rename(

    columns={'Q': 'Embarked_Q', 'S': 'Embarked_S', 'C': 'Embarked_C'})

y = data['Survived']

x = data.drop(['Sex', 'Embarked', 'Survived'], axis=1)

x.head()
# Mean age for nan values

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

x['Age'] = imp.fit_transform(x['Age'].values.reshape(-1, 1))

# Name handler for feature selection

name = lambda x: f'{x} features'

results = []

preprocessing = [("",lambda x: x), (" and scaled data", lambda x: MinMaxScaler().fit_transform(x)), (" and standardized data", lambda x: StandardScaler().fit_transform(x)), (" and normalized data", lambda x: Normalizer().fit_transform(x))]

for i in range(1, len(x.keys())): #with feature selection

    for j in preprocessing:

        rfe = RFE(svm.SVC(kernel='linear'), i)

        x_train, x_test, y_train, y_test = train_test_split(j[1](x), y, test_size=0.3)

        model = svm.SVC(kernel='linear')

        x_train = rfe.fit_transform(x_train, y_train)

        model.fit(x_train, y_train)

        results.append((name(i)+j[0], accuracy_score(y_test, model.predict(rfe.transform(x_test)))))

for j in preprocessing: #no feature selection

    model = svm.SVC(kernel='linear')

    x_train, x_test, y_train, y_test = train_test_split(j[1](x), y, test_size=0.3)

    model.fit(x_train, y_train)

    results.append((name(len(x.keys()))+j[0], accuracy_score(y_test, model.predict(x_test))))
for i in range(len(results)):

    print(f'Dataset {i} has {results[i][0]}, accuracy: {results[i][1]}')
plt.figure(figsize=(15,15))

plt.title("Accuracy of different datasets with a fixed svm linear kernel model")

for i in range(4):

    x_ind = [4*k+i for k in range(0, len(results)//4)]

    plt.plot(x_ind, list(map(lambda a: results[a][1], x_ind)), 'o-', linewidth=2, markersize=4, label={0:"no prep",1:"scaled",2:"standardized",3:"normalized"}[i])

plt.legend(loc='best')

plt.ylabel("Accuracy")

plt.xlabel("Dataset")

plt.show()