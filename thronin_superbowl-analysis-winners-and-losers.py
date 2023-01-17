# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')

df.head()
to_drop = ['MVP', 'SB', 'Stadium']

df.drop(to_drop, inplace=True, axis=1)

df.head()
df['Date'].is_unique
df['Date'] = pd.to_datetime(df['Date'])

df = df.set_index('Date')

df.head()
df.dtypes.value_counts()
df.isnull().sum()
new_names = {'Winner': 'Winning_Team',

             'Winner Pts': 'Winning_Pts',

             'Loser': 'Losing_Team',

             'Loser Pts': 'Losing_Pts'}

df.rename(columns=new_names, inplace=True)

df.head()
duplicateRowsDF = df[df.duplicated()]

duplicateRowsDF
df.describe(include='all')
df['Winning_Team'].value_counts().head(10).plot(kind='bar')
df['Winning_Pts'].plot(kind='box')
df['Losing_Team'].value_counts().head(10).plot(kind='bar')
df['Losing_Pts'].plot(kind='box')
df['State'].value_counts().head(10).plot(kind='bar')
# What teams have been to the SB the most, regardless if they won or lost?

df['Winning_Team'].append(df['Losing_Team']).value_counts().head(10).plot(kind='bar')
df['Winning_Pts'].plot(style='k.')
bin_edges = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

df['Winning_Pts'].hist(bins=bin_edges)
bin_edges = [0, 5, 10, 15, 20, 25, 30, 35]

df['Losing_Pts'].hist(bins=bin_edges)