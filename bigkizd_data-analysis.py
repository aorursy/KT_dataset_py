# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cereal.csv')

print(df.head())

print(df.info())

print(df.shape)

df['mfr'].value_counts(dropna=False)
sns.barplot(x='name', y='rating', data=df)
f, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.countplot(x='mfr', data=df, ax=axes[0])

sns.countplot(x='type', data=df, ax=axes[1])

sns.pairplot(data=df)
f, axes = plt.subplots(1, 2, figsize=(10, 5))

sns.scatterplot(x='mfr', y='weight', data=df, ax=axes[0])

sns.scatterplot(x='mfr', y='cups', data=df, ax=axes[1])
cereals = df.iloc[:, ~df.columns.isin(['name', 'mfr', 'type', 'rating'])].div(df.weight, axis=0)

cereals = pd.concat([df.iloc[:, df.columns.isin(['name', 'mfr', 'type', 'rating'])], cereals], axis=1)

cereals.head()
corr = df.iloc[:, ~cereals.columns.isin(['name', 'mfr', 'type'])].corr()
sns.heatmap(corr)
from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

columns = cereals.columns[3:]

cereals[columns] = scaler.fit_transform(cereals[columns])

cereals['Good'] = cereals.loc[:, ['protein', 'fiber', 'vitamins']].mean(axis=1)

cereals['Bad'] = cereals.loc[:, ['fat', 'solium', 'potass', 'sugars']].mean(axis=1)

sns.lmplot(x='Good', y='Bad', data=cereals)
cereals['new_ranking'] = cereals['Good']/cereals['Bad']
