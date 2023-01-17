# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

import matplotlib.pyplot as plt

import plotly

plotly.offline.init_notebook_mode()

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import numpy as np

import seaborn as sns

import calendar

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/911.csv")
df.head(3)
df.addr.unique().size
df=df.drop(['desc','addr'],axis=1)
df.dtypes
df['timeStamp']=pd.to_datetime(df.timeStamp)

df['year']=df['timeStamp'].dt.year

df['month']=df['timeStamp'].dt.month

df['quarter']=df['timeStamp'].dt.quarter

df['doy']=df['timeStamp'].dt.dayofyear

df['woy']=df['timeStamp'].dt.weekofyear

df['dow']=df['timeStamp'].dt.dayofweek

df['hour']=df['timeStamp'].dt.hour

df['minute']=df['timeStamp'].dt.minute

#df['year']=df['timeStamp'].dt.year

#df['year']=df['timeStamp'].dt.year
df=df.drop(['e'],axis=1)
df=df.drop(['timeStamp'],axis=1)
df.head()
sns.countplot(df['month'])
sns.countplot(df['quarter'])
sns.countplot(df['dow'])
sns.countplot(df['year'])
df.groupby(['title'])['title'].size().sort_values(ascending=False).head().plot(kind='pie',autopct='%1.1f%%')
df.groupby(['title'])['title'].size().sort_values(ascending=False).head().plot(kind='barh')
df.groupby(['title'])['title'].size().sort_values(ascending=True).head(7).plot(kind='pie',autopct='%1.1f%%')
df.groupby(['title'])['title'].size().sort_values(ascending=False).head(7).plot(kind='bar')
df.groupby(['zip']).size().sort_values(ascending=False).head(5).plot(kind='pie',autopct='%1.1f%%')
df.groupby(['twp']).size().sort_values(ascending=False).head(5).plot(kind='pie',autopct='%1.1f%%')
df.head()
df.groupby(['title','month'])['title'].size().sort_values(ascending=False).head(5).plot(kind='pie',autopct='%1.1f%%')
#TOP EMS Data

df[df['title'].str.contains("EMS")].groupby('title').size().sort_values(ascending=False).head().plot(kind='pie',autopct='%1.1f%%')
#TOP EMS Data

df[df['title'].str.contains("EMS")].groupby('month').size().sort_values(ascending=False).head().plot(kind='bar',color='y')