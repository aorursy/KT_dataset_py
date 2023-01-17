# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import random

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_full = pd.read_csv('../input/train.csv')

X_full.columns
ID = X_full.PassengerId 

NAME = X_full.Name #Maybe we could use this in the future, so let's store this here.

X_full.drop(['PassengerId','Name'], inplace=True, axis=1)

X_full.head()
X_full.describe(include='all')
X_full.isnull().sum()
plt.figure(figsize=(15,8))

sns.heatmap(X_full.isnull(), cbar=False)
X_full.drop(['Cabin','Ticket'], inplace=True, axis=1)

X_full.columns
X_full.drop(X_full[X_full.Embarked.isnull() == True].index, inplace=True)

plt.figure(figsize=(15,8))

sns.heatmap(X_full.isnull(), cbar=False)
plt.figure(figsize=(15,8))

sns.heatmap(X_full.corr(), cmap='magma', annot=True)
s = X_full['Age'].isnull()

X_full_age = X_full.copy() #Do this on a copy to see the distributions comparison







X_full_age.Age[s] = X_full_age.Age[s].map(lambda x: random.randrange(X_full_age.Age.median() - (X_full_age.Age.median()*0.25),

                                                       X_full_age.Age.median() + (X_full_age.Age.median()*0.25)))





plt.figure(figsize=(20,6))

sns.distplot(X_full_age.Age, ax=plt.subplot(1,2,1))

sns.distplot(X_full.Age.dropna(), ax=plt.subplot(1,2,2))

X_full_age['Age'] = pd.Series(pd.cut(X_full_age.Age,[0,10,20,30,40,50,60,70,80,90], labels= False, right=True))

X_full_age.head()
X_full_age.dtypes
X_full_age.Pclass = pd.Categorical(X_full_age.Pclass)

X_full_age = pd.get_dummies(X_full_age)

X_full_age.head()
X_full_age.isnull().sum()
X_full_age['Family'] = X_full_age.SibSp + X_full_age.Parch

X_full_age.drop(['SibSp','Parch'], axis=1, inplace=True)

X_full_age.head()
plt.figure(figsize=(15,8))

sns.heatmap(X_full_age.corr(), annot=True, cmap='magma')
ytrain = X_full_age.Survived

Xtrain = X_full_age.drop(['Survived'], axis=1)
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC



PGrid = {"C":[1,10,100],

        "gamma":[.01, 0.1],

        "kernel":['rbf'],

        "cache_size":[200,500,1000]}



model = SVC(random_state=1)

Xtrainsvm = StandardScaler().fit_transform(Xtrain)



gsearch = GridSearchCV(estimator=model, param_grid=PGrid, cv=5, iid=False)



score = cross_val_score(gsearch, X=Xtrainsvm, y=ytrain, cv=5)



score.mean()
Xtest = pd.read_csv('../input/test.csv')

ID = Xtest['PassengerId']

Xtest.drop(['PassengerId','Name','Cabin','Ticket'], inplace=True, axis=1)

plt.figure(figsize=(15,8))

sns.heatmap(Xtest.isnull(), cbar=False)
s = Xtest['Age'].isnull()



Xtest.Age[s] = Xtest.Age[s].map(lambda x: random.randrange(round(Xtest.Age.median() - (Xtest.Age.median()*0.25)),

                                                           round(Xtest.Age.median() + (Xtest.Age.median()*0.25))))



Xtest.isnull().sum()
s = Xtest['Fare'].isnull()



Xtest.Fare[s] = Xtest.Fare.mean()



Xtest.isnull().sum()
Xtest['Age'] = pd.Series(pd.cut(Xtest.Age,[0,10,20,30,40,50,60,70,80,90], labels= False, right=True))

Xtest.head()
Xtest.Pclass = pd.Categorical(Xtest.Pclass)

Xtest = pd.get_dummies(Xtest)

Xtest.head()
Xtest['Family'] = Xtest.SibSp + Xtest.Parch

Xtest.drop(['SibSp','Parch'], axis=1, inplace=True)

Xtest.head()
Xtest = StandardScaler().fit_transform(Xtest)
gsearch.fit(Xtrain, ytrain)



svmpredict = gsearch.predict(Xtest)
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score





xgb = XGBClassifier(n_estimators=1000, learning_rate=0.1, random_state=0)

xgbscore = cross_val_score(xgb, Xtrainsvm, ytrain, cv=5)

print(xgbscore.mean())

xgb.fit(Xtrainsvm, ytrain)

predictions = xgb.predict(Xtest)
submission = pd.concat([ID,pd.Series(predictions.tolist())], axis=1)

submission.columns = ['PassengerId','Survived']

submission.to_csv('predictions.csv', index=False)

submission.head()