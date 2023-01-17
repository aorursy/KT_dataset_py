import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_df.head()
titanic_df.info()
titanic_df.describe()
sns.factorplot('Sex',data=titanic_df, kind="count")
sns.factorplot('Sex',data=titanic_df, kind="count", hue = 'Pclass')
sns.factorplot('Pclass',data=titanic_df, kind="count", hue = 'Sex')
def male_female_child(passenger):

    age, sex = passenger

    if age < 16:

        return 'child'

    else:

        return sex
titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis = 1)
titanic_df[0:10]
sns.factorplot('Pclass',data=titanic_df, kind="count", hue = 'person')
titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['person'].value_counts()
fig = sns.FacetGrid(titanic_df, hue='Sex', aspect = 4)

fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim = (0, oldest))



fig.add_legend()
fig = sns.FacetGrid(titanic_df, hue='person', aspect = 4)

fig.map(sns.kdeplot, 'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim = (0, oldest))



fig.add_legend()