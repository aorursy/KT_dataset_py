# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output





# Load the Titanic datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full_data = [train, test]
train.head()

train.info()
train.isnull().sum(), test.isnull().sum()
train.shape, test.shape
train.Ticket.unique()
for data in full_data:

    data['FamSize'] = data['SibSp'] + data['Parch'] + 1



train[['FamSize','Survived']].groupby('FamSize').mean()
train.Embarked.unique()
train.Embarked.value_counts()
for data in full_data:

    data.Embarked.fillna('S',inplace = True)



train[['Embarked','Survived']].groupby('Embarked').mean()
for data in full_data:

    data.Fare.fillna(train.Fare.median(), inplace = True)



train['CategoricalFare'], bins = pd.qcut(train.Fare,4, retbins = True, labels = False)

test['CategoricalFare'] = pd.cut(test.Fare , bins=bins, labels=False, include_lowest=True)

train[['CategoricalFare','Survived']].groupby('CategoricalFare').mean()

for data in full_data:

    data.Age.fillna(train.Age.median(), inplace = True)

    

train['CategoricalAge'], bins = pd.qcut(train.Fare,6, retbins = True, labels = False)

test['CategoricalAge'] = pd.cut(test.Age , bins=bins, labels=False, include_lowest=True)

train[['CategoricalAge','Survived']].groupby('CategoricalAge').mean()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



for data in full_data:

    data['Title'] = data['Name'].apply(get_title)



pd.crosstab(train['Title'], train['Sex'])
train.sample(3)
for data in full_data:

    data.Title = data.Title.replace('Mlle', 'Miss')

    data.Title = data.Title.replace('Ms', 'Miss')

    data.Title = data.Title.replace('Mme', 'Mrs')

    data.Title[~data.Title.str.contains('Miss|Mrs|Mr|Master')] = 'Other'



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train.Cabin.unique()
def get_cabin(x):

    if isinstance(x, str) :

        c = x[0]

        if c in 'ABCDE':

            return c

    return np.nan

    



for data in full_data:

    data.Cabin = data.Cabin.apply(get_cabin)

    data.Cabin.fillna('NoCabin', inplace = True)

    

print (train[['Cabin', 'Survived']].groupby(['Cabin']).mean())
train.Title.unique()
train.sample(3)
train.Pclass.unique()
for data in full_data:

    #Mapping Sex

    data.Sex = np.where(data.Sex == 'male',1,0)

    #Mapping Sex

    data = pd.get_dummies(data, columns = ['Embarked','Pclass','Title','Cabin'])
train = pd.get_dummies(train, columns = ['Embarked','Pclass','Title','Cabin'])

test = pd.get_dummies(test, columns = ['Embarked','Pclass','Title','Cabin'])
columns_to_drop = ['PassengerId','Name', 'Ticket', 'SibSp','Parch','Age', 'Fare']

train = train.drop(columns_to_drop, axis = 1)

X_test = test.drop(columns_to_drop, axis = 1)

ids = test['PassengerId']



X_train = train.drop(['Survived'], axis = 1)

y_train = train.Survived
X_train.shape, y_train.shape, X_test.shape
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

import xgboost as xgb

import scipy as sp







param_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }



grid_lr = GridSearchCV(LogisticRegression(penalty='l2'), param_lr, cv = 10)

grid_lr.fit(X_train, y_train)

print(grid_lr.best_score_)

print(grid_lr.best_params_)



clf_lr = grid_lr.best_estimator_

# Save CV predictions for stacking

X_train_meta_lr = cross_val_predict(clf_lr, X_train, y_train, cv=30)



# Predict Test data for stacking

X_test_meta_lr = clf_lr.predict(X_test)



# set parameter grid for Random Forest Classifier

param_rf = {

                'max_features': [3,6,9, None], 

                'criterion': ['entropy', 'gini'],

                'max_depth': [3, 5, None], 

                'min_samples_split': [2, 3, 5],

                'min_samples_leaf': [1, 3],

                'bootstrap': [True, False]

             }



grid_rf = GridSearchCV(RandomForestClassifier(n_estimators = 100), param_rf, cv = 10)

grid_rf.fit(X_train, y_train)



print(grid_rf.best_score_)

print(grid_rf.best_params_)



clf_rf = grid_rf.best_estimator_

# Save CV predictions for stacking

X_train_meta_rf = cross_val_predict(clf_rf, X_train, y_train, cv = 30)



# Predict Test data for stacking

X_test_meta_rf = clf_rf.predict(X_test)



# set parameter grid for Gradient Boosting Classifier

param_gb = {

                'max_features': [3, 5, 7, None], 

                'max_depth': [3, 5, None], 

                'min_samples_split': [2, 3, 5],

                'min_samples_leaf': [1, 3],

                'loss' : ['deviance', 'exponential']

             }



grid_gb = GridSearchCV(GradientBoostingClassifier(n_estimators = 100), param_gb, cv = 10)

grid_gb.fit(X_train, y_train)

# grid_search = grid_search.fit(X_train, y_train)

# print the average CV score of the best model

print(grid_gb.best_score_)

print(grid_gb.best_params_)



clf_gb = grid_gb.best_estimator_

# Save CV predictions for stacking

X_train_meta_gb = cross_val_predict(clf_gb, X_train, y_train, cv = 30)



# Predict Test data for stacking

X_test_meta_gb = clf_gb.predict(X_test)

from sklearn.svm import SVC



param_sv = {'C': sp.stats.expon(scale=10), 

            'kernel': ['rbf'], 

            'gamma': sp.stats.expon(scale=0.1)}



random_sv = RandomizedSearchCV(SVC(), param_distributions=param_sv, n_iter=100, random_state = 273)

random_sv.fit(X_train, y_train)



print(random_sv.best_score_)

print(random_sv.best_params_)



clf_sv = random_sv.best_estimator_



# Save CV predictions for stacking

X_train_meta_sv = cross_val_predict(clf_sv, X_train, y_train, cv=30)



# Predict Test data for stacking

X_test_meta_sv = clf_sv.predict(X_test)



X_train_meta = pd.DataFrame([X_train_meta_lr, X_train_meta_gb, X_train_meta_sv, X_train_meta_rf]).transpose()

X_train_meta = pd.concat([X_train,X_train_meta], axis = 1)

X_test_meta = pd.DataFrame([X_test_meta_lr, X_test_meta_gb, X_test_meta_sv, X_test_meta_rf]).transpose()

X_test_meta = pd.concat([X_test,X_test_meta], axis = 1)



X_train_meta.shape, X_test_meta.shape
from sklearn.svm import SVC



param_sv = {'C': sp.stats.expon(scale=1), 

            'kernel': ['rbf','linear'], 

            'gamma': sp.stats.expon(scale=0.1)}



random_sv_meta = RandomizedSearchCV(SVC(), param_distributions=param_sv, n_iter=100, random_state = 373, cv = 10)

random_sv_meta.fit(X_train_meta, y_train)



print(random_sv_meta.best_score_)

print(random_sv_meta.best_params_)



clf_meta = random_sv_meta.best_estimator_

y_test = clf_meta.predict(X_test_meta)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': y_test })

output.to_csv('titanic-predictions.csv', index = False)