# libraries for data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# libraries for visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# scikit-learn machine learning libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_df=pd.read_csv('../input/train.csv');

test_df=pd.read_csv('../input/test.csv');

combine=[train_df,test_df];
# this will print features

print(train_df.columns.values)
#preview the data

train_df.head()
#information about dataset

train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean()
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean()
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
g=sns.FacetGrid(train_df,col="Survived")

g.map(plt.hist,"Age",bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train_df=train_df.drop(['Ticket','Cabin'],axis=1)

test_df=test_df.drop(['Ticket','Cabin'],axis=1)

combine=[train_df,test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

#OUTPUT-('After', (891, 10), (418, 9), (891, 10), (418, 9))
for dataset in combine:

    dataset['Title']=dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_df['Title'],train_df['Sex'])   