# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/database.csv')
df.head()
fig,ax = plt.subplots(figsize=(6,4))

sns.countplot(df.gender.dropna(),ax=ax)

plt.title('Gender Count')
fig,ax = plt.subplots(figsize=(6,4))

sns.distplot(df.age.dropna(),ax=ax)

plt.title('manner of death Count')
armed_df = df['armed'].value_counts().sort_values(ascending=False)[:15].reset_index()

fig,ax = plt.subplots(figsize=(6,4))

sns.barplot(x=armed_df['index'],y=armed_df['armed'],ax=ax)

ticks = plt.setp(ax.get_xticklabels(),rotation=90)

plt.title('Top 10 Armed')

plt.xlabel('Armed Category')

plt.ylabel('Total Amount of Armed')
fig,ax = plt.subplots(figsize=(6,4))

sns.countplot(df.race.dropna(),ax=ax)

plt.title('race of the victims Count')
df.race.value_counts()
date_series = pd.to_datetime(df['date'])

date_series_wd = [None]*len(date_series)

for ii in range(len(date_series)):

    date_series_wd[ii] = date_series[ii].dayofweek

print(date_series_wd)
df['day_of_week'] = pd.Series(date_series_wd, index=df.index)

df.head()
fig,ax = plt.subplots(figsize=(6,4))

sns.countplot(df.day_of_week.dropna(),ax=ax)

plt.title('day of the week')
fig,ax = plt.subplots(figsize=(6,4))

sns.countplot(y='state', data = df,ax=ax)

plt.title('state-wise')