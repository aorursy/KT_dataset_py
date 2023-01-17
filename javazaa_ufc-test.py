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
train_df = pd.read_csv('../input/ufcdata/data.csv')

test_df = pd.read_csv('../input/ufcdata/preprocessed_data.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()

train_df.describe(include=['O'])
train_df[['R_wins', 'no_of_rounds']].groupby(['R_wins'], as_index=False).mean().sort_values(by='no_of_rounds', ascending=False)
g = sns.FacetGrid(train_df, col='no_of_rounds')

g.map(plt.hist, 'R_wins', bins=20)
grid = sns.FacetGrid(train_df, col='no_of_rounds', row='R_win_by_KO/TKO', size=2.2, aspect=1.6)

grid.map(plt.hist, 'R_wins', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='R_win_by_KO/TKO', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'R_wins', 'R_age', 'R_Weight_lbs', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='R_win_by_KO/TKO', col='no_of_rounds', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'R_wins', 'R_age', alpha=.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['R_wins', 'R_age'], axis=1)

test_df = test_df.drop(['R_wins', 'R_age'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
X_train = train_df.drop("no_of_rounds", axis=1)

Y_train = train_df["no_of_rounds"]

X_test  = test_df.drop("R_win_by_KO/TKO", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape