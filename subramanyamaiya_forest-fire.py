# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # Visualization

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(dirname, filename),engine='python')

df.head()
df.info() 
df.isnull().sum()
# Converting Date column to datetime

df['date'] = pd.to_datetime(df['date'])
# Since date column values are already present in year and month feature, we can drop the date column

df.drop('date',axis=1,inplace=True)

df.head()
# Checking the unique year. i.e to get from and till which year the data is collected.

df['year'].unique()
# From 1998 till 2017, the data is collected.
df['state'].unique()
df['number'].value_counts()
df['number'] = df['number'].apply(lambda x: round(x))
# Checking no. of incidents recorded on each year.

df.groupby(by='year').sum().plot(kind='line',figsize=(12,8))

plt.xticks(df['year'].unique())
# Which state is facing this problem more?

df.groupby(by='state').sum()['number'].sort_values(ascending=True).plot(kind='bar')
# Checking the number of fire rates in each month

df.groupby(by='month').sum()['number'].plot(kind='bar')
# Lets see the maximum number of forestfire recorded in a month. 

df[df['number'] == df['number'].max()]
pd.crosstab(df['year'],df['state'],values=df['number'],aggfunc='sum').plot.bar(stacked=True,figsize=(20, 10))

plt.legend()
pd.crosstab(df['month'],df['state'],values=df['number'],aggfunc='sum').plot.bar(stacked=True,figsize=(20, 10))
pd.crosstab(df['month'],df['state'],values=df['number'],aggfunc='sum').T