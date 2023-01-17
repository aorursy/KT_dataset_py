import pandas as pd
import numpy as np
import seaborn as sns
% matplotlib inline
df = pd.read_csv('../input/data.csv')
df.info()
df.isnull().any()
df[df['Video Uploads']=='--']
t = df.replace('--', np.nan)
t['Grade'] = t['Grade'].astype('category')
t['Grade']=t['Grade'].cat.codes
t['Grade']
t=t.drop(columns="Channel name")
for i in range(1,5001):
    t.iloc[i-1,0]=i
t.head()
t.isnull().any()
t['Video Uploads']=t['Video Uploads'].fillna(0)
t['Video Uploads']=t['Video Uploads'].astype('uint64')
t['Subscribers'] = [x.replace('--','0') for x in t['Subscribers']]
t['Subscribers']=t['Subscribers'].astype('uint64')
t.info()
t.hist()
t.hist(figsize=(10,10))
t['Grade'].value_counts().plot(kind='bar')
t['Grade'].value_counts().plot(kind='pie',figsize=(8,8))
pd.plotting.scatter_matrix(t,figsize=(15,15))
t['Video views'].plot(kind='box');
sns.heatmap(t)
sns.heatmap(t.corr(),cmap = 'RdGy')
sns.pairplot(t.iloc[:,2:])
