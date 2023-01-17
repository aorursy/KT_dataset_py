# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score



import os

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train, test]
print(train.columns.values)
# preview the data

train.head()
train.describe()
train.describe(include=['O'])
train.info()
test.info()
pclass_surv = sns.factorplot(x = "Pclass",y = 'Survived',data = train,kind = "bar")
sex_surv = sns.factorplot(x = "Sex",y = 'Survived',data = train,kind = "bar")
sib_surv = sns.factorplot(x = "SibSp",y = 'Survived',data = train,kind = "bar")
parch_surv = sns.factorplot(x = "Parch",y = 'Survived',data = train,kind = "bar")
age_surv = sns.FacetGrid(train, col = 'Survived')

age_surv.map(sns.distplot, 'Age', bins = 20)
train = train.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)

test = test.drop(['Ticket', 'Cabin'], axis=1)

combine = [train, test]
print(train.shape, test.shape, combine[0].shape, combine[1].shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#combine["Title"] = combine["Title"].map({"Master":0, "Miss":1, "Mrs" : 2 , "Rare":3})

#combine["Title"] = combine["Title"].astype(int)

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
g = sns.factorplot(x = "Title", y = "Survived", data = train, kind = "bar")
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
train = train.drop(['Name'], axis=1)

test = test.drop(['Name'], axis=1)

combine = [train, test]

train.shape, test.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train.head()
train.info()
test.info()
fullDataset = pd.concat((train, test))

train['Age'] = train['Age'].fillna(fullDataset['Age'].mean())

test['Age'] = test['Age'].fillna(fullDataset['Age'].mean())
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(train.Embarked.dropna().mode()[0]) # most common value 'S'
test["Fare"] = test["Fare"].fillna(fullDataset["Fare"].mean())

train["Fare"] = train["Fare"].fillna(fullDataset["Fare"].mean())
test.info()
train.info()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.preprocessing import StandardScaler
headers_train = X_train.columns

headers_test = X_test.columns
train.head()
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
pd.DataFrame(X_train, columns=headers_train).head()
pd.DataFrame(X_test, columns=headers_test).head()
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

decision_tree_score = round(decision_tree.score(X_train, Y_train), 4)

decision_tree_score
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

linear_svc_score = round(linear_svc.score(X_train, Y_train), 4)

linear_svc_score
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)

random_forest_score = round(random_forest.score(X_train, Y_train), 4)

random_forest_score
# Gaussian Naive Bayes



cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 )

accuracies = cross_val_score(GaussianNB(), X_train, Y_train, cv  = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))



bayes_score = round(accuracies.mean(), 4)

bayes_score
# Support Vector Machines



C = [0.1, 1,1.5]

gammas = [0.001, 0.01, 0.1]

kernels = ['rbf', 'poly', 'sigmoid']

param_grid = {'C': C, 'gamma' : gammas, 'kernel' : kernels}



cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=8)



grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv)

grid_search.fit(X_train,Y_train)
print(grid_search.best_score_)

print(grid_search.best_params_)

print(grid_search.best_estimator_)
svm_grid = grid_search.best_estimator_

svm_score = round(svm_grid.score(X_train,Y_train), 4)

svm_score
# k-Nearest Neighbors algorithm



k_range = range(1,31)

weights_options=['uniform','distance']

param = {'n_neighbors':k_range, 'weights':weights_options}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)



grid.fit(X_train,Y_train)
print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
knn_grid= grid.best_estimator_

knn_score = round(knn_grid.score(X_train,Y_train), 4)

knn_score
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
results = pd.DataFrame({

    'Model': ['Decision Tree', 'Linear SVC', 'Random Forest', 'Gaussian Naive Bayes', 'Support Vector Machines', 'k-Nearest Neighbors algorithm'],

    'Score': [decision_tree_score, linear_svc_score, random_forest_score, bayes_score, svm_score, knn_score]})

results.sort_values(by = 'Score', ascending = False)
Y_pred = logreg.predict(X_test)#random_forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission1.csv', index=False)