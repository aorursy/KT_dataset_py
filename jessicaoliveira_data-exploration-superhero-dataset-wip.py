# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/superhero-set"))



# Any results you write to the current directory are saved as output.
df_information = pd.read_csv("../input/superhero-set/heroes_information.csv")

df_information.head()

df = df_information.copy()
print(df.shape)

df.info()
df.isnull().sum()
df.isnull().sum().sum()/df.shape[1]
# this way even empty strings are counted as null 

df.replace('', np.nan).isnull().sum()

df.replace('[-|]', np.nan, regex=True).isnull().sum()
print(df[df['Height'] < 0].shape)

df[df['Weight'] < 0].shape
df.loc[df['Height'] < 0] = np.nan

df.loc[df['Weight'] < 0] = np.nan
df.duplicated().sum()
print(df['name'].duplicated().sum())

print(df[df['name'].duplicated(keep=False)]['name'].unique())

print(df[df[['name']].duplicated(keep=False)])
print(df[['name', 'Gender', 'Publisher']].duplicated().sum())

df[df[['name', 'Gender', 'Publisher']].duplicated(keep=False)]
df.describe()
df_num = df.select_dtypes(include=['float64', 'int64'])

df_obj = df.select_dtypes(include=['object'])
print(df['Gender'].unique())

df['Gender'].value_counts()
import seaborn as sns

sns.countplot(df['Gender'])
sns.distplot(df['Height'], bins=10, kde=False)
sns.distplot(df['Weight'], bins=10, kde=False)