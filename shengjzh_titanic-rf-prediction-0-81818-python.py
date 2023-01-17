import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Loading Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full = pd.concat([train, test])
full.describe(include='all')
# missing value

full.isnull().sum()
age_bins = range(0, 80, 10)

train['Age_group'] = pd.cut(train['Age'], bins=age_bins)



facet = sns.FacetGrid(train, hue="Survived", size=5, aspect=2)

facet.map(sns.countplot, 'Age_group')

facet.add_legend()
sns.countplot(x='Sex', hue='Survived', data=train)
sex_mean_survived = train[['Sex', 'Survived']].groupby('Sex').mean()

print(sex_mean_survived)
facet_sex = sns.FacetGrid(train, col='Sex', hue='Survived', size=6)

facet_sex.map(sns.countplot, 'Age_group')

facet_sex.add_legend()
pclass_mean_survived = train[['Pclass', 'Survived']].groupby('Pclass').mean()

print(pclass_mean_survived)
facet_sex = sns.FacetGrid(train, col='Sex', hue='Survived')

facet_sex.map(sns.countplot, 'Pclass')

facet_sex.add_legend()
facet_sex = sns.FacetGrid(train, col='Pclass', row='Sex', hue='Survived')

facet_sex.map(sns.countplot, 'Age_group')

facet_sex.add_legend()