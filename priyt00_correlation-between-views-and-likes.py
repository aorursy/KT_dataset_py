# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling

from pandas.plotting import scatter_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=[]

col_list = ['title', 'views', 'likes']

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.csv'):

            df1=pd.DataFrame(pd.read_csv(os.path.join(dirname, filename),header=0,usecols=col_list))

            df1['country']=filename[:2]

            

            df.append(df1)

train=pd.concat(df,axis=0,ignore_index=True)
train.shape,train.isna().sum()

train.head(100)
train.describe()
train.corr()
sns.heatmap(train.corr(),annot=True)
figure=plt.figure(figsize=(10,10))

scatter_matrix(train)

plt.show()

#plt.title('Scatterplot matrix')

#plt.legend()
figure=plt.figure(figsize=(10,10))

sns.regplot(x='views',y='likes',data=train)

plt.title('Correlation between views and likes')
countrywise=pd.DataFrame(train.groupby(by=['country']).sum())

#countrywise.h

countrywise.index
#plt.subplots(1,2)

sns.barplot(x=countrywise.index,y=countrywise['likes'])

plt.show()

sns.barplot(x=countrywise.index,y=countrywise['views'])

plt.show()
titlewise=pd.DataFrame(train.groupby(by=['title']).sum())

titlewise.sort_values(by=['views','likes'],ascending=False,inplace=True)

#titlewise.head()
titlewise[titlewise['views']==titlewise['views'].max()]

titlewise[titlewise['likes']==titlewise['likes'].max()]
sns.barplot(x=titlewise.index[:10],y=titlewise.likes[:10])

plt.xticks(rotation=90)