# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

import os
file_dir = '../input/citibikeny'
!ls $file_dir | wc -l
files = glob.glob(os.path.join(file_dir, "*.csv"))
dataframes = []

for file in files:

    df = pd.read_csv(file)

    df = df.sample(frac = .005)

    dataframes.append(df)
# Concatenate all the dataframes into one

df = pd.concat(dataframes, ignore_index=True)
df.shape
df.dtypes
df['starttime'] = df['starttime'].astype('datetime64[ns]')

df['stoptime'] = df['stoptime'].astype('datetime64[ns]')
df['start date'] =  pd.to_datetime(df['starttime']).dt.date

df['start date'] =  pd.to_datetime(df['starttime']).dt.date

df['stop date'] =  pd.to_datetime(df['stoptime']).dt.date

df['start hour'] =  pd.to_datetime(df['starttime']).dt.hour

df['stop hour'] =  pd.to_datetime(df['stoptime']).dt.hour

df['month'] =  pd.to_datetime(df['stoptime']).dt.month

df['day'] =  pd.to_datetime(df['stoptime']).dt.day



df['start day_of_week'] = df['starttime'].dt.dayofweek

df['stop day_of_week'] = df['stoptime'].dt.dayofweek
df.head(5)
sns.distplot(df['month'])

plt.show()
sns.distplot(df['start hour'])

plt.show()
sns.catplot(x="day", y="start hour", hue="usertype", kind="swarm", data=df);
g = sns.relplot(x="start day_of_week", y="tripduration", kind="line",hue="usertype", data=df)

g.fig.autofmt_xdate()
g = sns.relplot(x="stop day_of_week", y="tripduration", kind="line",hue="usertype", data=df)

g.fig.autofmt_xdate()
g = sns.FacetGrid(df, col="start day_of_week", height=4, aspect=.6)

g.map(sns.barplot, "usertype", "tripduration");
n = 10

df['start station name'].value_counts()[:n].index.tolist()
print(df['start hour'].value_counts())

print(df['start day_of_week'].value_counts())
sns.relplot(x="start hour", y="start day_of_week", hue="usertype",data=df);
#0: unknown, 1: male, 2:female

sns.catplot(data=df,x="gender", kind="count")

plt.show()
#0: unknown, 1: male, 2:female

sns.catplot(data=df,x="usertype", kind="count", hue="gender")

plt.show()
sns.catplot(data=df,x="age", kind="count",hue="gender")

plt.show()
ax = sns.countplot(x="month", hue="usertype", data=df)
sns.catplot(x="start station name", kind="count", data=df);