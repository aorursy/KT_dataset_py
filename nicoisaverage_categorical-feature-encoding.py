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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

import scipy

from sklearn.linear_model import LogisticRegression

import optuna

from sklearn import base
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

submission = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv', index_col='id')



Ytrain = train['target']





print(train.shape)

print(test.shape)
summary = train.info()

summary
data = pd.concat([train, test], sort=False)

dummies = pd.get_dummies(data, columns=data.columns, sparse=True)

dummies = dummies.sparse.to_coo()

dummies = dummies.tocsr()



Xtrain = dummies[:len(train)]

Xtest = dummies[len(train):]
kf=StratifiedKFold(n_splits=10)



def objective(trial):

    C = trial.suggest_loguniform('C', 10e-10, 10)

    model = LogisticRegression(C=C, class_weight='balanced',max_iter=1000, solver='lbfgs', n_jobs=-1)

    score = cross_val_score(model, Xtrain, Ytrain, cv=kf, scoring='roc_auc').mean()

    return score
study=optuna.create_study()

study.optimize(objective, n_trials=100)

#print(study.best_params)

#print(study.best_value)

model = LogisticRegression(C=.0036, class_weight='balanced', verbose=0, max_iter=1000,

                          solver='lbfgs')

model.fit(Xtrain, Ytrain)

preds = model.predict_proba(Xtest)[:,1]

submission['target']=preds

submission.to_csv('ohcatty9.csv')
