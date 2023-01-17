# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use(u'ggplot')
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['reasons']=df['title'].apply(lambda x:x.split(':')[0])
df['reasons'].value_counts().head(3)
sns.countplot(x='reasons',data=df)
df['timeStamp'].dtype
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
time = df['timeStamp'].iloc[0]
time.hour
df['hour'] = df['timeStamp'].apply(lambda x : x.hour)
df['month'] = df['timeStamp'].apply(lambda x : x.month)
df['day of week'] = df['timeStamp'].dt.dayofweek
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day of week'] = df['day of week'].map(dmap)
plt.figure(figsize=(10,6))
sns.countplot('day of week',data=df,hue='reasons',palette='Set1')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.figure(figsize=(10,6))
sns.countplot(x='month',data=df,hue='reasons',palette='Set1')
plt.legend(bbox_to_anchor=(1, 1), loc=2)
byMonth = df.groupby(by=df['month']).count()
byMonth.head()
df['month'].groupby(by=df['month']).count().plot()
# create a new column in byMonth dataframe
byMonth['count'] = df['month'].value_counts()
byMonth['month'] = byMonth.index
# reset index from month to something else keeping previous index as a column
byMonth.reset_index(drop=True)
sns.lmplot(x='month',y='count',data=byMonth,fit_reg=True)
df['date'] = df['timeStamp'].apply(lambda x :  x.date())
plt.figure(figsize=(12,4))
df['month'].groupby(by=df['date'],sort=True).count().plot()
plt.figure(figsize=(12,5))
df[df['reasons']=='Traffic']['month'].groupby(by=df['date']).count().plot()
plt.title('Traffic')
plt.ylim(0,600)
plt.figure(figsize=(12,5))
df[df['reasons']=='Fire']['month'].groupby(by=df['date']).count().plot()
plt.title('Fire')
plt.ylim(0,180)
plt.figure(figsize=(12,4))
df[df['reasons']=='EMS']['month'].groupby(by=df['date']).count().plot()
plt.title('EMS')
plt.ylim(0,250)
df2 = df[['day of week','hour']]
df2 = df2.groupby(['day of week','hour']).size()
df2 = df2.unstack(level=-1)
df2
plt.figure(figsize=(12,6))
sns.heatmap(data=df2,vmin=300,vmax=3000,cmap='viridis',cbar=True)
sns.clustermap(data=df2,figsize=(15,10))
df3 = df[['day of week','month']]
df3 = df3.groupby(['day of week','month']).size()
df3 = df3.unstack(level=-1)
df3
plt.figure(figsize=(12,6))
sns.heatmap(data=df3,cmap='viridis',vmin=1000,vmax=5000)
sns.clustermap(data=df3,figsize=(15,10))
