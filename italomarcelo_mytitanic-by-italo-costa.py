# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#  Data Tabulated

from tabulate import tabulate

# ML Algoritmos

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.dummy import DummyRegressor
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
def eda(dataset, titulo='EDA'):

    print(f'=={titulo}==')

    print('INFO \n')

    print(tabulate(dataset.info(), headers='keys', tablefmt='psql'))

    print('\nHEAD \n', tabulate(dataset.head(), headers='keys', tablefmt='psql'))

    print('\nTAIL \n', tabulate(dataset.tail(), headers='keys', tablefmt='psql'))

    print('\nDESCRIBE \n', tabulate(dataset.describe(), headers='keys', tablefmt='psql'))

    print('\nEXAMPLES \n', tabulate(dataset.sample(5), headers='keys', tablefmt='psql'))

    print('\nNULL QTY \n', tabulate([dataset.isnull().sum()], headers=dataset.columns, tablefmt='psql'))

    print('\nSHAPE \n', tabulate([dataset.shape], headers=['ROWS', 'COLS'], tablefmt='psql'))
# eda in train dataset

eda(dataset=train, titulo='EDA [ data train ]')
# eda in test dataset

eda(dataset=test, titulo='EDA [ data test ]')
# data calculated

# rate women survived

w = train.query("Sex == 'female' and Survived == 1")

totw = train.loc[train.Sex == 'female']

ws = len(w) * 100 / len(totw)



# rate men survived

m = train.query("Sex == 'male' and Survived == 1")

totm = train.loc[train.Sex == 'male']

ms = len(m) * 100 / len(totm)



# rate people survived

tp = len(train)

ts = (len(m) + len(w)) * 100 / tp
print(f'Titanic Data Passengers:\n\n'

      f'Total: {tp}\n'

      f'Survives: {len(m) + len(w)}\n'

      f'Rate Survive: {ts:.2f}')
print(f'\nPassenger Female:\n'

      f'Total: {len(totw)}\nSurvives: {len(w)}'

      f'\nRate Survive: {ws:.2f}')
print(f'\nPassenger Male:\n'

      f'Total: {len(totm)}\nSurvives: {len(m)}'

      f'\nRate Survive: {ms:.2f}')
# target

y = train['Survived']

# Features

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
# Data dummy extract

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
# regressors list

print('\nAnalisando regressores:')

alg = []

score = []

regressors = [

        DecisionTreeRegressor(),

        RandomForestRegressor(),

        SVR(),

        LinearRegression(),

        GradientBoostingRegressor(),

        PoissonRegressor(),

        DummyRegressor(),

        LogisticRegression(),

        GaussianNB()

    ]

for regressor in regressors:

    model = regressor

    model.fit(X, y)

    score.append(model.score(X, y))

    alg.append(regressor)



bestML = pd.DataFrame(columns=['Regressor', 'Score'])

bestML['Regressor'] = alg

bestML['Score'] = score

bestML = bestML.sort_values(by='Score', ascending=False)

print(tabulate(bestML, headers='keys', tablefmt='psql'))

# selecting best model

bestmodel = bestML.values[0][0]

print(f'Selected model: {bestmodel}\n')
# training 

bestmodel.fit(X, y) 

print('Model trained')

# predict survive

predict = bestmodel.predict(X_test)

print(f'Predict created. \nTarget: Survived')
# show predicts

predict
# creating a dictionary

mydict = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predict})
# creating CSV File - content: mydict

mydict.to_csv('my_submission_ItaloCosta.csv', index=False)
print('Score: ', bestmodel.score(X, y))