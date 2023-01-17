# data analysis and wranling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

#import matplotlib as plt

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
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_' * 40)

test_df.info()
train_df.describe()

# train_df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
train_df.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', row='Survived')

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.Facetgrid(train_df, col='Embarked')

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
#grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', 

#                     palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2,

                    aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape,

      combine[0].shape, combine[1].shape)

train_df_wrangle = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df_wrangle = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine_wrangle = [train_df_wrangle, test_df_wrangle]



"After", train_df_wrangle.shape, test_df_wrangle.shape, combine_wrangle[0].shape, combine_wrangle[1].shape
for dataset in combine_wrangle:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_df_wrangle['Title'], train_df_wrangle['Sex'])
for dataset in combine_wrangle:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt',\

                                                'Col', 'Don', 'Dr', 'Major',\

                                                'Rev', 'Sir', 'Jonkheer',\

                                                'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_df_wrangle[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine_wrangle:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

train_df_wrangle.head()
train_df_wrangle2 = train_df_wrangle.drop(['Name', 'PassengerId'], axis=1)

test_df_wrangle2 = test_df_wrangle.drop(['Name'], axis=1)

combine_wrangle2 = [train_df_wrangle2, test_df_wrangle2]

train_df_wrangle2.shape, test_df_wrangle2.shape
for dataset in combine_wrangle2:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)



train_df_wrangle2.head()
#grid = sns.FacetGrid(train_df_wrangle2, col='Pclass', hue='Sex')

grid = sns.FacetGrid(train_df_wrangle2, row='Pclass', col='Sex', size=2.2, 

                     aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()