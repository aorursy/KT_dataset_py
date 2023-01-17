# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Import math libraries

import numpy as np

import pandas as pd



# Import ML algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



# Import data prep libraries

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.model_selection import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Import viz libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

from matplotlib import cm

import seaborn as sns
# get train and test datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



full = train.append(test)
cmap = sns.diverging_palette( 220 , 5 , as_cmap = True )

sns.heatmap(train.corr(), annot=True, cmap=cmap)
ax = sns.kdeplot(train[train['Survived']==1]['Age'], shade=True, label='Survived')

ax = sns.kdeplot(train[train['Survived']==0]['Age'], shade=True, label='Perished')

ax.set(xlim=(0, train['Age'].max()))
facet = sns.FacetGrid(train, hue='Survived', aspect=4, row='Sex')

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()
facet = sns.FacetGrid(train, hue='Survived', aspect=3)

facet.map(sns.kdeplot, 'Fare', shade=True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()
sns.boxplot(x="Survived", y="Fare", data=train[train["Fare"] < 400])
facet = sns.FacetGrid(train)

facet.map(sns.barplot, 'Embarked', 'Survived')

facet.add_legend()
facet = sns.FacetGrid(train)

facet.map(sns.barplot, 'Sex', 'Survived')

facet.add_legend()
facet = sns.FacetGrid(train)

facet.map(sns.barplot, 'Pclass', 'Survived')

facet.add_legend()
facet = sns.FacetGrid(train)

facet.map(sns.barplot, 'SibSp', 'Survived')

facet.add_legend()
facet = sns.FacetGrid(train)

facet.map(sns.barplot, 'Parch', 'Survived')

facet.add_legend()
full["FamSize"] = full["SibSp"] + full["Parch"] + 1

full["Alone"] = full["FamSize"].map(lambda x: 1 if x == 1 else 0)

full["SmallFamily"] = full["FamSize"].map(lambda x: 1 if 1 < x < 5 else 0)

full["LargeFamily"] = full["FamSize"].map(lambda x: 1 if x >= 5 else 0)
full.head()