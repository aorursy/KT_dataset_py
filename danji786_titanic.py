# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from learntools.core import *

import random as rnd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import warnings

#warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





def get_mae_leaf(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)



def get_mae_depth(max_depth, best_leaf, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(max_depth=max_depth, max_leaf_nodes=best_leaf, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
train_file_path='../input/train.csv'

train_data = pd.read_csv(train_file_path).set_index('PassengerId')

train_data.columns



test_file_path='../input/test.csv'

test_data = pd.read_csv(test_file_path).set_index('PassengerId')
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Fare', bins=20)
g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_data, test_data]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_data['Title'], train_data['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'RM')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Mme', 'Mlle'], 'RF')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train_data['TMr']=train_data['Title'].apply(lambda row: 1 if row == 'Mr' else 0) 

train_data['TMiss']=train_data['Title'].apply(lambda row: 1 if row == 'Miss' else 0) 

train_data['TMrs']=train_data['Title'].apply(lambda row: 1 if row == 'Mrs' else 0) 

train_data['TMaster']=train_data['Title'].apply(lambda row: 1 if row == 'Master' else 0) 

train_data['TRM']=train_data['Title'].apply(lambda row: 1 if row == 'RM' else 0) 

train_data['TRF']=train_data['Title'].apply(lambda row: 1 if row == 'RF' else 0) 



test_data['TMr']=test_data['Title'].apply(lambda row: 1 if row == 'Mr' else 0) 

test_data['TMiss']=test_data['Title'].apply(lambda row: 1 if row == 'Miss' else 0) 

test_data['TMrs']=test_data['Title'].apply(lambda row: 1 if row == 'Mrs' else 0) 

test_data['TMaster']=test_data['Title'].apply(lambda row: 1 if row == 'Master' else 0) 

test_data['TRM']=test_data['Title'].apply(lambda row: 1 if row == 'RM' else 0) 

test_data['TRF']=test_data['Title'].apply(lambda row: 1 if row == 'RF' else 0) 





train_data = train_data.drop(['Title'], axis=1)

test_data = test_data.drop(['Title'], axis=1)

combine = [train_data, test_data]



train_data.head()
train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)
train_data = train_data.drop(['Sex'], axis=1)

test_data = test_data.drop(['Sex'], axis=1)



combine = [train_data, test_data]

train_data.head()

test_data['Age'].fillna(test_data['Age'].dropna().median(), inplace=True)

train_data['Age'].fillna(train_data['Age'].dropna().median(), inplace=True)
freq_port = train_data.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data['EmbarkedC']=train_data['Embarked'].apply(lambda row: 1 if row == 'C' else 0) 

train_data['EmbarkedQ']=train_data['Embarked'].apply(lambda row: 1 if row == 'Q' else 0) 

train_data['EmbarkedS']=train_data['Embarked'].apply(lambda row: 1 if row == 'S' else 0) 



test_data['EmCbarkedC']=test_data['Embarked'].apply(lambda row: 1 if row == 'C' else 0) 

test_data['EmbarkedQ']=test_data['Embarked'].apply(lambda row: 1 if row == 'Q' else 0) 

test_data['EmbarkedS']=test_data['Embarked'].apply(lambda row: 1 if row == 'S' else 0) 



train_data = train_data.drop(['Embarked'], axis=1)

test_data = test_data.drop(['Embarked'], axis=1)
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

train_data['Fare'].fillna(train_data['Fare'].dropna().median(), inplace=True)
train_y=train_data['Survived']

train_X=train_data.drop(['Survived'], axis=1)
from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestClassifier(n_estimators=100), train_X, train_y, cv=10, n_jobs=-1, scoring='neg_mean_squared_error').mean()
#rf_train=RandomForestRegressor(max_depth=best_depth, max_leaf_nodes=best_leaf,random_state=0)

rf_train=RandomForestClassifier(n_estimators=100)

rf_train.fit(train_X, train_y)



final_v=rf_train.predict(test_data)

print(final_v)
output = pd.DataFrame({'PassengerId': test_data.index,

                      'Survived': final_v})

output.to_csv('submission.csv', index=False)