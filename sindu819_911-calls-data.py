import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
df=pd.read_csv('../input/911.csv')
df['zip'].value_counts().head()
df['twp'].value_counts().head()
len(df['title'].unique())
df['Reasons']=df['title'].apply(lambda x:x.split(':')[0])
df['Reasons'].head()
k=df.Reasons.value_counts().head(3)
k
sns.countplot('Reasons',data=df,palette='viridis')

type(df['timeStamp'].iloc[0])
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['timeStamp'].iloc[0]


df['Hour']=df.timeStamp.apply(lambda x:x.hour)
df['Month']=df.timeStamp.apply(lambda x:x.month)
df['day of week']=df.timeStamp.apply(lambda x:x.dayofweek)
df['day of week'].head()


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day of week']=df['day of week'].map(dmap)
df['day of week'].head()


sns.countplot(x='day of week',data=df,hue='Reasons',palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
piv=df.groupby(['day of week'])['Hour'].value_counts().unstack()
piv.head()


pivm=df.groupby(['day of week'])['Month'].value_counts().unstack()
pivm.head()
plt.figure(figsize=(12,6))
sns.heatmap(pivm,cmap='viridis')


