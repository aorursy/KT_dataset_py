# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')
combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()

train_df.tail()
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Type','Fee','Age','Breed1', 'Breed2', 'Gender', 'Color1','Color2','Vaccinated','Dewormed','Sterilized','State','AdoptionSpeed']].groupby(['State'], as_index=False).mean().sort_values(by='Type', ascending=False)
train_df[["Breed1", "AdoptionSpeed"]].groupby(['Breed1'], as_index=False).mean().sort_values(by='AdoptionSpeed', ascending=False)
g = sns.FacetGrid(train_df, col='Type')
g.map(plt.hist, 'AdoptionSpeed', bins=10)
grid = sns.FacetGrid(train_df, col='Type', row='Age', height=2.2, aspect=2.5)
grid.map(plt.hist, 'AdoptionSpeed', alpha=.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Type', height=2.2, aspect=10.2)
grid.map(sns.pointplot, 'Age', 'AdoptionSpeed', 'Breed1', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Breed1', col='AdoptionSpeed', height=2.2, aspect=10.2)
grid.map(sns.barplot, 'Age', 'Fee', alpha=.5, ci=None)
grid.add_legend()
