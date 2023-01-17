import numpy as np

import pandas as pd

import os

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
X_train = pd.read_csv('../input/titanic/train.csv')

X_test = pd.read_csv('../input/titanic/test.csv')

X_all = pd.concat([X_train, X_test], axis = 0)
X_train
print(X_all.isnull().sum().sort_values(ascending = False))

print(X_all.shape)

X_all.info()
g = sns.FacetGrid(X_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
X_train = X_train.drop(['Cabin'], axis = 1)

X_test = X_test.drop(['Cabin'], axis = 1)
X_train = X_train.drop(['Ticket'], axis = 1)

X_test = X_test.drop(['Ticket'], axis = 1)
X_all = [X_train, X_test]
for data in X_all:

    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
X_train[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean()
pd.crosstab(X_train['Title'], X_train['Sex'])
for data in X_all:

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')
pd.crosstab(X_train['Title'], X_train['Sex'])
X_train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean()
title_mapping = {"Mr": 1, "Other": 2, "Master": 3, "Miss": 4, "Mrs": 5} #ordered from least to most likely to survive

for data in X_all:

    data['Title'] = data['Title'].map(title_mapping)
X_all[0] = X_all[0].drop(['Name'], axis = 1)

X_all[1] = X_all[1].drop(['Name'], axis = 1)
s_mapping = {'female': 1, 'male': 0}

for d in X_all:

    d['Sex'] = d['Sex'].map(s_mapping).astype(int)
X_all[0]
X_all[0][['Embarked', 'Survived']].groupby(['Embarked'], as_index = False).mean()
for data in X_all:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
X_all[0][['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean()
c = X_all[0].columns

plt.figure(figsize=(14,12))

g = sns.heatmap(X_all[0][c].corr(),annot=True, cmap = "coolwarm")
def agegroup(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = (0, 1, 2, 3, 4, 5, 6, 7)#['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df

agegroup(X_all[0])

agegroup(X_all[1])
X_all[0]
def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = [0, 1, 2,3, 4]#['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df

simplify_fares(X_all[0])

simplify_fares(X_all[1])
X_all[0] = X_all[0].drop(['Embarked'], axis = 1)

X_all[1] = X_all[1].drop(['Embarked'], axis = 1)
X_all[0]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



y_train_all = X_all[0]['Survived']

X_train_all = X_all[0].drop(['Survived', 'PassengerId'], axis=1)



num_test = 0.20

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_train_all, y_train_all, test_size=num_test, random_state=2019)
clf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



acc_scorer = make_scorer(accuracy_score)



grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(Xtrain, Ytrain)



clf = grid_obj.best_estimator_



clf.fit(Xtrain, Ytrain)
predictions = clf.predict(Xtest)

print(accuracy_score(Ytest, predictions))
from sklearn.model_selection import KFold



def run_kfold(clf):

    n = 4

    kf = KFold(n_splits=n)

    kf.get_n_splits(X_train_all)

    outcomes = []

    fold = 0

    for train_index, test_index in kf.split(X_train_all):

        fold+=1

        Xtrain1, Xtest1 = X_train_all.iloc[train_index], X_train_all.iloc[test_index]

        Ytrain1, Ytest1 = y_train_all.iloc[train_index], y_train_all.iloc[test_index]

        clf.fit(Xtrain1, Ytrain1)

        predictions = clf.predict(Xtest1)

        accuracy = accuracy_score(Ytest1, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(clf)
ids = X_all[1]['PassengerId']

predictions = clf.predict(X_all[1].drop('PassengerId', axis=1))





output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('titanic-predictions.csv', index = False)

output.head()
print(y_train_all.iloc[0])

print(X_train_all.iloc[0])