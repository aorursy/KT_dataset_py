# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import re

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



data = pd.read_csv('../input/fifa-20-complete-player-datasets/FIFAdatas.csv')

df = pd.DataFrame(data)

print(df.head())

# Any results you write to the current directory are saved as output.
print(df.shape)
print(df.describe())
print(df.info())
print(df['Team'].value_counts().head(10))
print(df['Position'].value_counts())
plt.plot(df['Position'].value_counts())

plt.xlabel('Positions')

plt.ylabel('Number of Players')

plt.title('Number of Player at each position')

plt.show()
def Age_num(df_age):

    try:

        age = float(df_age[2:-2])

    except ValueError:

        age = NaN

    return age   

df['Age_Num'] = df['Age'].apply(Age_num)

print(df['Age_Num'])
plt.figure(figsize = (16,9))

AOax = sns.violinplot(x = 'Age_Num', y = 'Overall', data = df)

AOax.set_title('Relation between the Age and Overall')

plt.xticks(Rotation = 70)

plt.show()
def Value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]

        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value

df['Value_float'] = df['Value'].apply(Value_to_int)

df['Value_float'] = df['Value_float']/1e8

print('Top 10 Clubs with all players combined Value')

print(df.groupby('Team')['Value_float'].sum().sort_values(ascending = False).head(10))
def Wage_to_float(df_wage):

    try:

        wage = float(df_wage[1:-1])

        suffix = df_wage[-1:]

        if suffix == 'K':

            wage = wage * 1000

    except ValueError:

        wage = 0

    return wage   

df['Wages'] = df['Wage'].apply(Wage_to_float)

print('Top 10 Clubs with all players combined Wage')

print(df.groupby('Team')['Wages'].sum().sort_values(ascending = False).head(10))
print('Least 10 Team with Market Value')

print(df.groupby('Team')['Value_float'].sum().sort_values(ascending = True).head(10))
plt.figure(figsize = (16,9))

x = df['Age_Num']

Agex = sns.distplot(x , bins = 60 , kde = True ,color = 'g')

Agex.set_title('Numbers of players according to their age')

Agex.set_xlabel('Age')

Agex.set_ylabel('Numbers of Players')

plt.show()