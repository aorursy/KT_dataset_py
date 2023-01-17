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
from tabulate import tabulate

import seaborn as sns, matplotlib.pyplot as plt

import warnings

# ML Algoritmos

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor

from sklearn.svm import SVR

from sklearn.naive_bayes import GaussianNB

from sklearn.dummy import DummyRegressor



# ML selecao de dados de treino e teste

from sklearn.model_selection import train_test_split

# calcular o menor erro medio absoluto entre 2 dados apresentados

from sklearn.metrics import mean_absolute_error



warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
def eda(dataset, title='EDA'):

    print(f'=={title}==')

    print('INFO \n')

    print(tabulate(dataset.info(), headers='keys', tablefmt='psql'))

    print('\nHEAD \n', tabulate(dataset.head(), headers='keys', tablefmt='psql'))

    print('\nTAIL \n', tabulate(dataset.tail(), headers='keys', tablefmt='psql'))

    print('\nDESCRIBE \n', tabulate(dataset.describe(), headers='keys', tablefmt='psql'))

    print('\n5 SAMPLES \n', tabulate(dataset.sample(5), headers='keys', tablefmt='psql'))

    print('\nNULLS QTY \n', dataset.isnull().sum().sum())

    print('\nSHAPE \n', tabulate([dataset.shape], headers=['rows', 'cols'], tablefmt='psql'))
# Exploration Data Analysis

eda(dataset=train, title='EDA [ train ]')

eda(dataset=test, title='EDA [ train ]')
def repNaN(df):

    ncols = df.select_dtypes(include=['int64', 'float64']).columns

    for i in ncols:

        df[i] = df[i].fillna(df[i].min())

    ccols = df.select_dtypes(include=['object']).columns

    for i in ccols:

        df[i] = df[i].fillna('-')

    return df
print(train.isnull().sum().sum(), train.shape)

print(test.isnull().sum().sum(), test.shape)
train = repNaN(train)

test = repNaN(test)
print(train.isnull().sum().sum(), train.shape)

print(test.isnull().sum().sum(), test.shape)
# set target and features variables

def correlacao(df, varT, xpoint=-0.5, showGraph=True):

    corr = df.corr()

    print(f'\nFeatures correlation:\n'

          f'Target: {varT}\n'

          f'Reference.: {xpoint}\n'

          f'\nMain features:')

    if showGraph:

        sns.heatmap(corr,

                    annot=True, fmt='.2f', vmin=-1, vmax=1, linewidth=0.01,

                    linecolor='black', cmap='RdBu_r'

                    )

        plt.title('Correlations between features w/ target')

        plt.show()



    corrs = corr[varT]

    features = []

    for i in range(0, len(corrs)):

        if corrs[i] > xpoint and corrs.index[i] != varT:

            print(corrs.index[i], f'{corrs[i]:.2f}')

            features.append(corrs.index[i])

    return features

# best targets correlations

varTarget = 'SalePrice'

varFeatures = correlacao(df=train, varT=varTarget, xpoint=0.5, showGraph=False)
print(f'Target: {varTarget}\n'

      f'Features: {list(varFeatures)}')
# start prediction

y = train[varTarget]

X = pd.get_dummies(train[varFeatures])

X_test = pd.get_dummies(test[varFeatures])



print(y.shape, X.shape, X_test.shape)
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



print('Choose the Regressor [0..8]: ', end='')

# case you wanna the alg

# ml = int(input())

# model = bestML.values[ml][0]

ml = DecisionTreeRegressor()
# Train.. predict and save submission



model = ml

print(f'Selected model: {model}\n')



print(f'Training model with {model}...')

model.fit(X, y)

print('Model trained')



print('Starting predict with test data')

predict = model.predict(X_test)

print(f'Score: {model.score(X, y):.2f}')

print(f'Predict created. \nTarget: {varTarget}')



print('Creating data submission')

mydict = pd.DataFrame({'Id': test.Id, 'SalePrice': predict})

mydict.to_csv('hp_submission_ItaloCosta.csv', index=False)

print('Submission created.')