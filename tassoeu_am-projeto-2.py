# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysing and wrangling

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_1')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_2')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_2')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_3')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_4')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_5')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_6')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_7')
g = sns.FacetGrid(train_df, col='target')

g.map(plt.hist, 'feat_8')