import pandas as pd
% matplotlib inline
df = pd.read_csv('../input/data.csv')
df.head()
df[df['Video Uploads']=='--']
import numpy as np
t = df.replace('--', np.nan)
t['Grade'] = t['Grade'].astype('category')
t['Grade']=t['Grade'].cat.codes
t=t.drop(columns="Channel name")
t['Video Uploads']=t['Video Uploads'].fillna(0)
t['Video Uploads']= t['Video Uploads'].astype('uint64')
t['Subscribers']=[x.replace('--','0') for x in t['Subscribers']]
t['Subscribers']=t['Subscribers'].astype('uint64')
for i in range(1,5001):
    t.iloc[i-1,0]=i
t.info()
t.hist(figsize=(10,10))
t['Grade'].value_counts().plot(kind='bar')
t['Grade'].value_counts().plot(kind='pie',figsize=(15,15))
t['Video views'].plot(kind='box')
pd.plotting.scatter_matrix(t,figsize=(10,10))
import seaborn as sns
sns.heatmap(t.corr())