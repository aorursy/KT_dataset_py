# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/montcoalert/911.csv")
df.info()
df.head()
len(df['title'].unique())
x=df['title'].iloc[0]

x.split(':')[0]
df['Reason']=df['title'].apply(lambda title:title.split(':')[0])

df['Reason']
df['Reason'].value_counts()
sns.countplot(x='Reason',data=df)
type(df['timeStamp'].iloc[0])

df["timeStamp"]=pd.to_datetime(df['timeStamp'])
time=df['timeStamp'].iloc[0]

df['hour']=df['timeStamp'].apply(lambda time:time.hour)
df['month']=df['timeStamp'].apply(lambda time:time.month)

df['day of week']=df['timeStamp'].apply(lambda time:time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['day of week'] = df['day of week'].map(dmap)
sns.countplot(x='day of week',data=df,hue='Reason')
sns.countplot(x='month',data=df,hue='Reason')
t=df['timeStamp'].iloc[0]

df['date']=df['timeStamp'].apply(lambda t:t.date())
df.groupby('date').count()['lat'].plot()

plt.tight_layout()