# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from speedml import Speedml
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]

# reading in csv files and combining data sets for machine learning
sml = Speedml('../input/train.csv', 

              '../input/test.csv', 

              target = 'Survived',

              uid = 'PassengerId')

sml.shape()

# here I am inputting the csv files via speedml
sml.train.head()

# looking to visualize the head of my data table
sml.train.tail()

# visualizing the end of my data table
print(train_df.columns.values)

# here are the features that I could potentially use in my algorithm
train_df.info()

print('_'*40)

test_df.info()

# basic information about the data that I have under each feature
train_df.describe()

# better visualization of the data
sml.plot.correlate()

# This is a very important line of code that I took from Manav's kaggle

# The heat map shows the correlation between every feature and survivability
sml.feature.drop(['Embarked'])
sml.feature.drop(['Cabin'])
sml.feature.drop(['Ticket'])
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# I initially thought of this feature idea, but the code I used is courtesy of Manav Seghal

# this code is from his speedml notebook

sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')

sml.feature.add('FamilySize', 1)

sml.feature.drop(['SibSp', 'Parch'])
sml.plot.bar('FamilySize', 'Survived')

# simple bar plot to visualize family size vs survivability
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# male = 0 and female = 1

#for dataset in combine:

    #dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

#train_df.dtypes

sml.feature.drop('Sex')
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 45), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 64), 'Age'] = 2

    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
sml.plot.bar('Pclass', 'Survived')
sml.feature.drop(['Name'])
train_df.dtypes
sml.eda()
sml.feature.impute()
sml.plot.importance()

# plot of every feature's relative importance
sml.model.data()
select_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

fixed_params = {'learning_rate': 0.1, 'subsample': 0.8, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)
select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

fixed_params = {'max_depth': 3, 'min_child_weight': 1, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)
tuned_params = {'learning_rate': 0.1, 'subsample': 0.8, 

                'max_depth': 3, 'min_child_weight': 1,

                'seed':0, 'colsample_bytree': 0.8, 

                'objective': 'binary:logistic'}

sml.xgb.cv(tuned_params)
tuned_params['n_estimators'] = sml.xgb.cv_results.shape[0] - 1

sml.xgb.params(tuned_params)
sml.xgb.classifier()
sml.model.evaluate()

sml.plot.model_ranks()
sml.model.ranks()
sml.xgb.fit()

sml.xgb.predict()

sml.xgb.feature_selection()
sml.xgb.sample_accuracy()
sml.save_results(

    columns={ 'PassengerId': sml.uid,

             'Survived': sml.xgb.predictions }, 

    file_path='output/titanic-speedml.csv'.format(sml.slug()))

sml.slug()