# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
plt.scatter(train.OverallQual, train.SalePrice)
train['Condition1n2'] = train.Condition1 + '_' + train.Condition2
trainCond = train.drop(['Condition1', 'Condition2'], axis=1)
test_bench(trainCond)
train.groupby('Condition1n2')['SalePrice'].mean().plot.bar()
train_bench = pd.read_csv("../input/train.csv")
test_bench(train_bench)
from sklearn import linear_model

from sklearn.model_selection import train_test_split, GridSearchCV

def test_bench(train, seeds=range(3)):

    score = []

    for seed in seeds:

        train = train.fillna(train.mean())

        train_encoded = pd.get_dummies(train)

        X_train, X_val, y_train, y_val = train_test_split(

            train_encoded.drop('SalePrice', axis=1), 

            train_encoded.SalePrice, 

            test_size=0.3, random_state=seed)

        lm = linear_model.Lasso(normalize=True)

        params = {'alpha':[0.1, 0.3, 1, 3, 10]}

        cv = GridSearchCV(lm, param_grid=params, n_jobs=4)

        cv.fit(X= X_train, y=y_train)

        score.append(cv.score(X_val, y_val))

    print(sum(score) / len(seeds))