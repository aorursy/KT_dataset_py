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
titanic_df = pd.read_csv('../input/train.csv')
titanic_df.head()
titanic_df.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Check the gender

sns.factorplot(x='Sex', data = titanic_df, kind = 'count')
# Check gender by classes

sns.factorplot(x='Pclass', data = titanic_df, kind = 'count', hue = 'Sex')
# Classify sex to male, female and child at the age of 16

def male_female_child(passenger):

    age, sex = passenger

    

    if age <16:

        return 'child'

    else:

        return sex
titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)
titanic_df
sns.factorplot(x='Pclass', data = titanic_df, kind = 'count', hue = 'person')
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['person'].value_counts()
# Male and female distribution by age

fig = sns.FacetGrid(titanic_df,hue='Sex', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
# Male, female and child distribution by age

fig = sns.FacetGrid(titanic_df,hue='person', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
# Class distribution by age

fig = sns.FacetGrid(titanic_df,hue='Pclass', aspect=4)

fig.map(sns.kdeplot, 'Age', shade=True)



oldest = titanic_df['Age'].max()



fig.set(xlim=(0,oldest))



fig.add_legend()
titanic_df.head()

# Organize cabin information by dropping NaN values first

deck = titanic_df['Cabin'].dropna()
deck.head()
from pandas import Series, DataFrame

levels = []

for level in deck:

    levels.append(level[0])

    

cabin_df = DataFrame(levels)

cabin_df.columns = ['Cabin']

sns.factorplot(x='Cabin', data=cabin_df, kind = 'count', palette='winter_d')
cabin_df = cabin_df[cabin_df.Cabin !='T']

sns.factorplot(x='Cabin', data=cabin_df, kind='count')
titanic_df.head()
titanic_df['Alone'] = titanic_df['SibSp'] + titanic_df['Parch']

titanic_df['Alone'].head()
titanic_df['Alone'].loc[titanic_df['Alone']>0]='With Family'

titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'
titanic_df.head()
sns.factorplot(x = 'Alone', data = titanic_df,kind = 'count', palette = 'Blues')
titanic_df['Survivor']=titanic_df.Survived.map({0:'no', 1:'yes'})

titanic_df.head()
sns.factorplot(x = 'Survivor', data = titanic_df, kind = 'count', palette = 'Set1')
sns.factorplot('Pclass', 'Survived', hue ='person', data = titanic_df)
sns.lmplot('Age', 'Survived', data = titanic_df)
sns.lmplot('Age', 'Survived', hue = 'Pclass', data =titanic_df, palette = 'winter')
generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue = 'Pclass', data =titanic_df, palette = 'winter', x_bins=generations)
sns.lmplot('Age', 'Survived', hue = 'Sex', data =titanic_df, palette = 'winter')