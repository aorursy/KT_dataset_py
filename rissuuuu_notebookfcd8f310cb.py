# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pl

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv('../input/stacklite/questions.csv')

df2=pd.read_csv('../input/stacklite/question_tags.csv')
df1.head()
df2.head()
df1.shape
df2.shape
df1['Id'].nunique(),df2['Id'].nunique()
df1=df1.merge(df2,left_on='Id',right_on='Id')
df1.isnull().sum()
df1.isnull().sum()/len(df1)
df1['Created_year']=pd.DatetimeIndex(df1['CreationDate']).year
df1['Created_month']=pd.DatetimeIndex(df1['CreationDate']).month
df1.head(10)
pl.figure(figsize=(10,6))

sns.countplot(df1['Created_year'])

pl.show()
pl.figure(figsize=(10,8))

ax=sns.countplot(y=df1['Created_month'])

pl.yticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

for i,j in enumerate(df1['Created_month'].value_counts().reset_index().sort_values('index')['Created_month']):

    ax.text(10,i,j)

pl.show()
df1['Created_month'].value_counts().reset_index().sort_values('index')['Created_month']
top20tags=df1['Tag'].value_counts().reset_index().head(20)
top20tags.columns=['Tag','Counts']

top20tags
pl.figure(figsize=(10,10))

ax=sns.barplot(x='Counts',y='Tag',data=top20tags)

for i,j in enumerate(top20tags['Counts']):

    ax.text(100000,i,j)
def top20Tags(data):

    if data in tags:

        return data

    else:

        return 'Ignore'

    

    

    

tags=list(top20tags['Tag'])

print(tags)
df1['Tag']=df1['Tag'].apply(lambda x: top20Tags(x))
df1=df1[df1['Tag']!='Ignore']
df1.shape
plot_group=df1.groupby(['Tag','Created_year'])['Created_year'].count()
plot_group.plot(kind='barh',figsize=(10,50))

pl.show()
plot_group=df1.groupby(['Created_year','Tag'])['Created_year'].count()

plot_group
plot_group.plot(kind='barh',figsize=(10,50))

pl.show()
sns.heatmap(df1.corr(),annot=True)
df1['Dayofweek']=pd.DatetimeIndex(df1['CreationDate']).dayofweek
df1.groupby(['Dayofweek','Tag'])['Tag'].count().plot(kind='barh',figsize=(10,40))

pl.show()
df1.groupby(['Tag','Dayofweek'])['Tag'].count().plot(kind='barh',figsize=(10,40))

pl.show()