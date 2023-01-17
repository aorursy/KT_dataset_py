from __future__ import division



import numpy as np

import pandas as pd

import matplotlib.pyplot as p

import seaborn as sns

sns.set_style('white')

sns.set_palette('muted')
train = pd.read_csv('../input/train.csv')
train.head(10)
print(train.isnull().sum())

print("\nTotal: %i" % len(train['PassengerId']))
test = pd.read_csv('../input/test.csv')

print(test.isnull().sum())

print("\nTotal: %i" % len(test['PassengerId']))
train.describe()
surv = train[train['Survived'] == 1]

nsurv = train[train['Survived'] == 0]

print(len(nsurv) / (len(surv) + len(nsurv)))
f = p.figure(figsize=[12,8])

f.add_subplot(241)

sns.barplot('Pclass', 'Survived', data=train)

f.add_subplot(242)

sns.barplot('SibSp', 'Survived', data=train)

f.add_subplot(243)

sns.barplot('Parch', 'Survived', data=train)

f.add_subplot(244)

sns.barplot('Sex', 'Survived', data=train)

f.add_subplot(245)

sns.barplot('Embarked', 'Survived', data=train)

f.add_subplot(246)

sns.distplot(surv['Age'].dropna().values, bins=range(0, 80, 5), kde=False, color='blue', norm_hist=True,

            label='Survived')

sns.distplot(nsurv['Age'].dropna().values, bins=range(0, 80, 5), kde=False, color='red', norm_hist=True,

             axlabel='Age', label='Died')

p.legend()

f.add_subplot(247)

sns.distplot(np.log10(surv['Fare'].dropna().values+1), bins=np.arange(0,3,.1875), kde=False, color='blue',

             norm_hist=True, label='Survived')

sns.distplot(np.log10(nsurv['Fare'].dropna().values+1), bins=np.arange(0,3,.1875), kde=False, color='red',

             norm_hist=True,axlabel=r'log$_{10}$(Fare)', label='Died')

p.legend()

p.tight_layout()