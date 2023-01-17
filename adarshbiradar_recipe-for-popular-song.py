import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
data.head()
print("The shape of the data is ",data.shape)
print("The columns are: ")

print(data.columns)
print(data['Unnamed: 0'])
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.sort_values(axis=0,ascending=False,inplace=True,by='Popularity')
data.head()
data.describe()
data.isnull().sum()
plt.figure(figsize=(10,10))

correlation=data.corr(method='spearman')

plt.title('Correlation heatmap')

sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,center=1)
track_name=data['Track.Name'].value_counts()

track_name[:10]
track_name[39:49]
artist_name=data['Artist.Name'].value_counts()

artist_name[:20]
plt.figure(figsize=(18,9))

sns.barplot(x=artist_name[:20],y=artist_name[:20].index)

plt.title("Artist name")
genre=data['Genre'].value_counts()

genre
plt.figure(figsize=(18,9))

sns.barplot(x=genre[:10],y=genre[:10].index)

plt.title("Genre")
plt.figure(figsize=(12,8))

sns.regplot(x='Beats.Per.Minute', y='Popularity',ci=None, data=data)

sns.kdeplot(data['Beats.Per.Minute'],data.Popularity)

plt.title("BPM and Popularity")
beats=data['Beats.Per.Minute']

print("min :",beats.min())

print("max :",beats.max())

print("mean :",beats.mean())
plt.figure(figsize=(12,8))

sns.regplot(x='Energy', y='Popularity',ci=None, data=data)

sns.kdeplot(data.Energy,data.Popularity)
energy=data['Energy']

print("min :",energy.min())

print("max :",energy.max())

print("mean :",energy.mean())
plt.figure(figsize=(12,8))

sns.regplot(x='Danceability', y='Popularity',ci=None, data=data)

sns.kdeplot(data.Danceability,data.Popularity)
plt.figure(figsize=(12,8))

sns.regplot(x='Loudness..dB..', y='Popularity',ci=None, data=data)

sns.kdeplot(data['Loudness..dB..'],data.Popularity)
loudness=data['Loudness..dB..']

print("min :",loudness.min())

print("max :",loudness.max())

print("mean :",loudness.mean())
plt.figure(figsize=(12,8))

sns.regplot(x='Liveness', y='Popularity',ci=None, data=data)

sns.kdeplot(data['Liveness'],data.Popularity)
liveness=data['Liveness']

print("min :",liveness.min())

print("max :",liveness.max())

print("mean :",liveness.mean())
plt.figure(figsize=(12,8))

sns.regplot(x='Liveness', y='Popularity',ci=None, data=data)

sns.kdeplot(data['Liveness'],data.Popularity)
plt.figure(figsize=(12,6))

sns.distplot(data['Length.'])
length=data['Length.']

print('Max :',length.max())

print('Min :',length.min())

print("Mean :",length.mean())
arr=[x for x in range(100,321,20)]

arr=tuple(arr)

length_cat=pd.cut(length,arr)
length_counts=length_cat.value_counts()

print(length_counts)
plt.figure(figsize=(18,9))

sns.barplot(x=length_counts,y=length_counts.index)
plt.figure(figsize=(12,8))

sns.jointplot(x='Acousticness..', y='Popularity',kind='kde', data=data)
plt.figure(figsize=(12,8))

sns.regplot(x='Speechiness.', y='Popularity',ci=None, data=data)

sns.kdeplot(data['Speechiness.'],data.Popularity)
Speechiness=data['Speechiness.']

print("min :",Speechiness.min())

print("max :",Speechiness.max())

print("mean :",Speechiness.mean())