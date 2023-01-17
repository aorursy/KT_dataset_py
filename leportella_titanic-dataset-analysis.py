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
df = pd.read_csv('../input/train.csv')
type(df)
df.head(2)
df.shape
df.shape[0]
df.columns
df.describe()
df['Age'].describe()
df['Age'] > 10
children = df['Age'] <= 10

children.sum()
df['Age'].plot.hist()
df['Fare'].plot.hist(bins=50)
df['Survived'].value_counts()
df['Embarked'].value_counts()
df['Pclass'].value_counts()
class_count = df['Pclass'].value_counts()

graph = class_count.plot.pie()
survived_count = df['Survived'].value_counts()

graph = survived_count.plot.bar()
filter_by_survived = df['Survived'] == 1 

#filter_by_survived.head()
filtered_df = df[filter_by_survived]

#filtered_df.head()
class_count = filtered_df['Pclass'].value_counts()

graph = class_count.sort_index().plot.bar()
class_count.sort_index()
df['Sex'].value_counts()
314/(577+314)
sex_count = filtered_df['Sex'].value_counts()

graph = sex_count.plot.bar()
def age_to_ageclass(age):

    if age <= 10:

        return 'child'

    if age <= 45:

        return 'adult'

    else:

        return 'elderly'
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Ageclass'] = df['Age'].apply(age_to_ageclass)
df.head(10)
filtered_df = df[filter_by_survived]
age_count = filtered_df['Ageclass'].value_counts()

graph = age_count.plot.bar()
grouped = df.groupby(['Pclass', 'Survived']).size().unstack()
import seaborn 
seaborn.heatmap(grouped, cmap='bone_r')