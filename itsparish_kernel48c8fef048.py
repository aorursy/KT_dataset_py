

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")
df.head()
df.info()
df.describe(include='all')
df.shape
df.isnull().sum()/df.shape[0]*100
sns.countplot(df['awarders'])
df.drop(['awarders'],axis=1,inplace=True)
sns.countplot(df['over_18'])
df['over_18'].value_counts()
below18_score=df['score'].where(df['over_18']==False).mean()

plus18_score=df['score'].where(df['over_18']==True).mean()

print('below18_score:',below18_score)

print('plus18_score:',plus18_score)
df[['author','score']].groupby(['author'],as_index=False).mean().sort_values(by='score',ascending=False)
df['total_awards_received'].value_counts()
df_1=df[['total_awards_received','score']].groupby(['total_awards_received'],as_index=False).mean().sort_values(by='score',ascending=False)
df_1
sns.regplot(x='total_awards_received',y='score',data=df_1)