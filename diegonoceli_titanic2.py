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
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]
media=train_df[["Pclass", "Fare"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Fare', ascending=False)

media
mediaclasse0=media['Fare'][0].mean()

mediaclasse1=media['Fare'][1].mean()

mediaclasse2=media['Fare'][2].mean()
for dataset in combine:    

    dataset.loc[ dataset['Pclass'] == 1, 'Taxa'] = dataset['Fare']-mediaclasse0

    dataset.loc[ dataset['Pclass'] == 2, 'Taxa']= dataset['Fare']-mediaclasse1

    dataset.loc[ dataset['Pclass'] == 3, 'Taxa'] = dataset['Fare']-mediaclasse2

train_df.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
train_df['TaxaRange'] = pd.cut(train_df['Taxa'], 20)

train_df[['TaxaRange', 'Survived']].groupby(['TaxaRange'], as_index=False).mean().sort_values(by='TaxaRange', ascending=True)
grid = sns.FacetGrid(train_df, row='TaxaRange', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Survived', alpha=.5, bins=20)

grid.add_legend()
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Taxa', bins=20)