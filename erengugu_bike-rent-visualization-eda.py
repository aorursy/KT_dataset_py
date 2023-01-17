# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import calendar

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


df.head()
df.info()
df.isnull().sum()
sns.heatmap(df.isnull());
list(df.dtypes)
df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]

df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]

df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]

df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]

df['year'] = df['year'].map({2011:0, 2012:1})

df.head()
test["hour"] = [t.hour for t in pd.DatetimeIndex(test.datetime)]

test["day"] = [t.dayofweek for t in pd.DatetimeIndex(test.datetime)]

test["month"] = [t.month for t in pd.DatetimeIndex(test.datetime)]

test['year'] = [t.year for t in pd.DatetimeIndex(test.datetime)]

test['year'] = test['year'].map({2011:0, 2012:1})

test.head()
df.drop('datetime',axis=1,inplace=True)

df.head()
df.drop(['casual','registered'],axis=1,inplace=True)
df.season.value_counts()
df.weather.value_counts()
corr = df[["temp","atemp","humidity","windspeed","count"]].corr()

plt.figure(figsize=(12,9))

sns.heatmap(corr,annot=True,square=True,linewidths=.5,cmap="Greens")

plt.show()
plt.figure(figsize=(7,5))

sns.factorplot(x='holiday',data=df,kind='count',size=5,aspect=1)

plt.show()
plt.figure(figsize=(7,5))

sns.countplot(x='workingday',data=df)

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(data=df[['temp','atemp', 'windspeed', 'count']])
plt.figure(figsize=(12,6))

sns.barplot(x="hour",y="count",data=df);
plt.figure(figsize=(12,4))

sns.barplot(x="month",y="count",data=df);
plt.figure(figsize=(22,4))

sns.barplot(x="humidity",y="count",data=df);
plt.figure(figsize=(18,4))

sns.barplot(x="windspeed",y="count",data=df);
plt.figure(figsize=(18,4))

sns.distplot(df.windspeed);
plt.figure(figsize=(18,4))

sns.distplot(df.hour);
plt.scatter(x="temp",y="count",data=df,color="Green");
plt.scatter(x="hour",y="count",data=df,color="Green");
plt.figure(figsize=(10,6))

sns.factorplot(x="day",y='count',kind='bar',data=df);
df.head()
season=pd.get_dummies(df['season'],prefix='season')

df=pd.concat([df,season],axis=1)

season=pd.get_dummies(test['season'],prefix='season')

test=pd.concat([test,season],axis=1)



weather=pd.get_dummies(df['weather'],prefix='weather')

df=pd.concat([df,weather],axis=1)

weather=pd.get_dummies(test['weather'],prefix='weather')

test=pd.concat([test,weather],axis=1)
df.columns
test.head()
df.columns.to_series().groupby(df.dtypes).groups