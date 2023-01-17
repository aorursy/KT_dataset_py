# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.head()
print(df['Sex'].value_counts())

sns.countplot(x = 'Sex', data = df)
sns.countplot(x = 'Pclass', hue = 'Sex', data = df)
def child(passenger):

    age, sex = passenger

    

    if age < 18:

        return 'child'

    else:

        return sex
df['Male_female_child'] = df[['Age', 'Sex']].apply(child,axis = 1)

df.head(10)
sns.countplot(x = 'Pclass', hue = 'Male_female_child', data = df)
df['Age'].hist(bins = 80)
df['Age'].mean()
fig = sns.FacetGrid(data = df, hue = 'Male_female_child', aspect = 6)

fig.map(sns.kdeplot,'Age', shade = True)

fig.set(xlim = (0, df['Age'].max()))

fig.add_legend()
fig = sns.FacetGrid(data = df, hue = 'Sex', aspect = 6)

fig.map(sns.kdeplot,'Age', shade = True)

fig.set(xlim = (0, df['Age'].max()))

fig.add_legend()
fig = sns.FacetGrid(data = df, hue = 'Pclass', aspect = 6)

fig.map(sns.kdeplot,'Age', shade = True)

fig.set(xlim = (0, df['Age'].max()))

fig.add_legend()
sns.countplot(x = 'Embarked', data = df)
sns.countplot(x = 'Survived', data = df)
sns.countplot(x = 'Survived', hue = 'Male_female_child',data = df)
sns.countplot(x = 'Survived', hue = 'Pclass',data = df)
sns.violinplot(x = 'Survived', y = 'Fare', data = df)