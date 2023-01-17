# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt  # data viz

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import sklearn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

print("DataFrame shape:  ", train_df.shape)

train_df.head()
train_df.info()
train_df.describe()
# Displaying the frequency counts for each categorical feature/label

for col in ['Survived', 'Pclass', 'Embarked']:

    print(train_df[col].value_counts())
train_df.corr()['Survived'].sort_values()
sns.barplot(x='Sex', y='Survived', ci=None, data=train_df)
sns.catplot(x='Survived', y='Age', col='Sex', kind='violin', data=train_df)
pd.crosstab(train_df['Pclass'], train_df['Embarked'], train_df['Survived'], aggfunc='mean')
data = pd.crosstab(train_df['Pclass'], train_df['Embarked'], train_df['Survived'], aggfunc='mean')

sns.heatmap(data, center=0.5)
sns.violinplot(x='Pclass', y='Age', hue='Survived', split=True, data=train_df)
sns.catplot('Survived', col='Embarked', data=train_df, kind='count')
sns.catplot(x='Age', y='Pclass', hue='Survived', row='Embarked', data=train_df, orient='h', kind='violin')
sns.catplot(x='SibSp', y='Parch', col='Sex', hue='Survived', kind='swarm', dodge=True, height=8, aspect=1.75, data=train_df)
sns.barplot(x='SibSp', y='Survived', hue='Sex', ci=None, data=train_df)
sns.barplot(x='Parch', y='Survived', hue='Sex', ci=None, data=train_df)