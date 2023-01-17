# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

import datetime

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

%matplotlib inline
! cd ../input ;ls
df = pd.read_csv('../input/charlotte-hot-chocolate-15k-2019/Hot Chocolate 15K Results.csv')
df.head()
df[df.duplicated(['BIB'],False)]
ax = df.AGE.value_counts().reindex(['0-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-100']).plot.bar()

ax.set_ylabel('Pariticpant Count');
df.GENDER.value_counts().plot.bar()
df.PACE.dtype, df.TIME.dtype
df['PACE_SEC']= df.PACE.apply(lambda x: int(x.split(':')[0])*60+int(x.split(':')[1]))
df.head()
time_split = lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]) if int(x.split(':')[0])>2 else int(x.split(':')[0])*3600 + int(x.split(':')[1])*60 + int(x.split(':')[2])
df['TIME_SEC']= df.TIME.apply(time_split)
df.head()
ax = plt.figure(figsize=(10,5))

ax = df.groupby(['HOMETOWN']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))

ax.set_ylabel('Mean Pace (Seconds)');

ax.set_title('Mean Pace of Participants Hometown');
ax = plt.figure(figsize=(10,5))

ax = df.groupby(['HOMETOWN']).agg(np.mean)['TIME_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))

ax.set_ylabel('Mean Time (Seconds)');
ax = plt.figure(figsize=(10,5))

ax = df.groupby(['AGE']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=plt.cm.Blues_r(np.arange(15)))

ax.set_ylabel('Mean Pace (Seconds)');
ax = df.boxplot(column=['PACE_SEC'],by=['AGE'],figsize=(10,5));
ax = plt.figure(figsize=(10,5))

ax = df.groupby(['GENDER']).agg(np.mean)['PACE_SEC'].sort_values()[:15].plot.bar(color=['b','r'])

ax.set_ylabel('Mean Pace (Seconds)');
ax = df.PACE_SEC.plot.hist(bins=5)

ax.set_xlabel('Pace (Seconds)');

ax.set_ylabel('Count');
bins = list(np.histogram_bin_edges(df.PACE_SEC,bins=5,range=(df.PACE_SEC.min()-1,df.PACE_SEC.max()+1)))
bin_labels = ['UltraFast','Fast','Medium','Slow','Turtle']
df['PACE_GROUP'] = pd.cut(df['PACE_SEC'],bins=bins,labels=bin_labels)
df.head(5)
df['count'] = np.ones(len(df))
df.head()
gender = df.groupby(['PACE_GROUP','GENDER']).count()['count']
ax = plt.figure(figsize=(10,5));

ax = gender.unstack(1).plot.bar(rot=0,subplots=False)

ax.set_ylabel('Count');
age =  df.groupby(['PACE_GROUP','AGE']).count()['count']
ax = plt.figure(figsize=(10,5));

ax = age.unstack(0).plot.bar(rot=90,subplots=False,stacked=True)

ax.set_xlabel('Count');

ax.set_ylabel('Age Group');
rank = df.groupby(['AGE']).agg(min)['RANK']
ax = rank.reindex(['0-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-100']).plot.bar()

ax.set_ylabel('Min Rank')
pd.plotting.scatter_matrix(df[['RANK','PACE_SEC']],figsize=(10,10));
sns.jointplot(x=['RANK'], y=['PACE_SEC'], data=df,kind='kde');
pd.plotting.scatter_matrix(df[['RANK','BIB']],figsize=(10,10),diagonal='kde');
sns.jointplot(x=['RANK'], y=['BIB'], data=df,kind='kde');