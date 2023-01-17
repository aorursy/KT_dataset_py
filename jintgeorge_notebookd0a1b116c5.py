import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd



#visualizations

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import style

#%matplotlib inline

style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')





#print(train_df.columns.values)

#print(train_df.Ticket.head(6))

#train_df.info()

#print('_' * 40)

#print(test_df.info())

#print(train_df.describe())

#print(train_df.describe(include=['O']))

#train_df[['Age', 'Fare']].plot()

#plt.show()



#ML

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



train_df[['Pclass', 'Survived']].groupby('Pclass', as_index = False).mean().sort_values(by = 'Survived', ascending=False)

train_df[['Sex', 'Survived']].groupby('Sex', as_index = False).mean().sort_values(by = 'Survived', ascending=False)
train_df[['Sex', 'Survived']].groupby('Sex', as_index = False).mean().sort_values(by = 'Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby('SibSp', as_index = False).mean().sort_values(by = 'Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby('Parch', as_index = False).mean().sort_values(by = 'Survived', ascending=False)
g = sns.FacetGrid(train_df, col = 'Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col = 'Survived', row = 'Pclass', size=2.2, aspect = 1.6)

grid.map(plt.hist, 'Age', alpha = 0.5, bins = 20)

grid.add_legend()