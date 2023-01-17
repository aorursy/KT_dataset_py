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
import pandas as pd 

df = pd.DataFrame([

  [1, 'San Diego', 100],

  [2, 'Los Angeles', 120],

  [3, 'San Francisco', 90],

  [4, 'Sacramento', 115],

],

  columns=[

    'Store ID',

    'Location',

    'Number of Employees',

  ])



print(df)
df.to_csv('../input/new-csv-file.csv')
df = pd.read_csv("../input/train.csv")
df
df.head()
df.info()
df.shape
df.columns
df[['Name','Sex']].head(10)
df['Sex'].value_counts()
df['Pclass'].value_counts()
df['Age'].mean()
#max min mean of fare col

df['Fare'].max()
df['Fare'].min()
df['Fare'].mean()
#each embarked

df['Embarked'].value_counts()
len(df)
male_df = df[df['Sex']=='male']
len(male_df)
df_age_above_50 = df[(df['Age'] > 50) & (df['Sex']=='male')]
#no of m and f btw age 30 and 50

df_male_female_btw_age_30_and_50 = df[(50>df['Age']) & (df['Age']>30)]

df_male_female_btw_age_30_and_50
#no of f with embarked val as s

df_female_emarked_as_s = df[(df['Sex']=='female') & (df['Embarked']=='S')]

df_female_emarked_as_s
#select 1st row

df.iloc[0]
df.Location.isin(['San Diego','Los Angeles'])