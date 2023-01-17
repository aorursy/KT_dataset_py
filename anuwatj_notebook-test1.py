# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



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



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]



print(train_df.columns.values)



# preview the data

train_df.head()

#train_df.tail()



#train_df.info()

#print('_'*40)

#test_df.info()



#train_df.describe()

# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.

# Review Parch distribution using `percentiles=[.75, .8]`

# SibSp distribution `[.68, .69]`

# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`



#train_df.describe(include=['O'])



#train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)



# To list all unique values in column 'Sex'

#train_df.Sex.unique()



# not working

#train_df.pivot(index='Sex', columns='Pclass', values='Survived')



#pivot table

#train_df.pivot_table(values='Survived', index=['Sex'], columns=['Pclass'], aggfunc=np.sum)

#pivot table, age should be segragated into ranges 

#train_df.pivot_table(values='Survived', index=['Sex', 'Age'], columns=['Pclass'], aggfunc=np.sum)



#PLOTTING

#g = sns.FacetGrid(train_df, col='Survived')

#g.map(plt.hist, 'Age', bins=20)



# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

#grid.map(plt.hist, 'Age', alpha=.5, bins=20)

#grid.add_legend



# grid = sns.FacetGrid(train_df, col='Embarked')

#grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

#grid.add_legend()



# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

#grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

#grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

#grid.add_legend()



print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape









for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
