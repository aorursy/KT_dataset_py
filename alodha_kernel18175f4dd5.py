# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option('display.max_rows',50)
# Any results you write to the current directory are saved as output.
#loading the data
df = df_kick = pd.read_csv("../input/ks-projects-201801.csv",nrows=10000)
df.head()
# totla no. 0f records
df.shape
#no. of records that were pledged and met goal
df1 = df[df['pledged']>=df['goal']]
df1.shape


##we will loook at both df and df1 data frames parallely to understand the diference
df.describe()

print(df.info())
print(df.nunique())
#if we compare it with above values..we still have variety in our data
print(df1.nunique())
#now nalysis of our dataset
df['state'].value_counts().head(10).plot.bar()
#(df['state'].value_counts().head(10)/10000).plot.bar()
df['state'].value_counts()
df1['state'].value_counts()
#analysis on the basis of month to seee
from datetime import datetime
df['mont'] =  pd.to_datetime(df['launched']).dt.month
df1['mont'] =  pd.to_datetime(df1['launched']).dt.month


#grouping to see when are the maximum no. of projects launched
grouped = df.groupby(['mont'])
grouped.size()
grouped.size().plot(kind = 'bar')
plt.show()
grouped = df1.groupby(['mont'])
grouped.size().plot(kind ='bar')
plt.legend()
plt.show()
#could not deduce much from it though
#tabular data to see the overall suceess on the basis of month
clarity_color_table0 = pd.crosstab(index=df["mont"], columns=df["state"])
clarity_color_table0['succ'] = clarity_color_table0['successful']/df.groupby(['mont']).size()
clarity_color_table0
#tabular data to see the overall suceess on the basis of month
clarity_color_table0 = pd.crosstab(index=df1["mont"], columns=df1["state"])
clarity_color_table0['succ'] = clarity_color_table0['successful']/df1.groupby(['mont']).size()
clarity_color_table0
##looking at categories distribution

df['category'].value_counts()#.head(10).plot.bar()


df1['category'].value_counts()
#analysis of data on bassis of category  --better representation below using crosstb
#grouped = df.groupby(['category','state'])
#grouped.size()    # Figure size
clarity_color_table = pd.crosstab(index=df["category"], columns=df["state"])

clarity_color_table['succ'] = clarity_color_table['successful']/df.groupby(['category']).size()
clarity_color_table[clarity_color_table['succ']>=0.5]
clarity_color_table = pd.crosstab(index=df1["category"], columns=df1["state"])

clarity_color_table['succ'] = clarity_color_table['successful']/df1.groupby(['category']).size()
clarity_color_table[clarity_color_table['succ']<1]

