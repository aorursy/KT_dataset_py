# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv',encoding='ISO-8859-1')
df.head()
df.info()
df.describe().T
df.shape
df.isnull().sum()
df.drop('Unnamed: 0', axis =1, inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
df.rename(columns={'title':'song', 'artist':'artist', 'top genre':'genre', 'year':'year', 'bpm':'beats_per_minute','nrgy':'energy',
                    'dnce':'danceability','dB':'loudness','live':'liveness', 'val':'valence', 'dur':'length', 'acous':'acousticness',
                    'spch':'speechiness','pop':'popularity'}, inplace=True)
df.columns
df['genre'].nunique()
df['genre'].value_counts().head(10)
df['genre'].value_counts().head(10).plot.pie(figsize=(10,10),autopct='%1.1f%%')
plt.title('Top 10 genre of 2010-2019')
df['artist'].nunique()
df['artist'].value_counts().head(10)
df['artist'].value_counts().head(10).plot.bar()
fig,axes = plt.subplots(nrows=4, ncols=3, figsize=(14,10))
plt.tight_layout()
axes[0][0].plot(df['beats_per_minute'].value_counts().sort_index())
axes[0][0].set_title('bpm')
axes[0][1].plot(df['energy'].value_counts().sort_index())
axes[0][1].set_title('Energy')
axes[0][2].plot(df['danceability'].value_counts().sort_index())
axes[0][2].set_title('Danceability')

axes[1][0].plot(df['liveness'].value_counts().sort_index())
axes[1][0].set_title('liveness')
axes[1][1].plot(df['valence'].value_counts().sort_index())
axes[1][1].set_title('valence')
axes[1][2].plot(df['length'].value_counts().sort_index())
axes[1][2].set_title('length')

axes[2][0].plot(df['loudness'].value_counts().sort_index())
axes[2][0].set_title('loudness')
axes[2][1].plot(df['acousticness'].value_counts().sort_index())
axes[2][1].set_title('acousticness')

axes[3][0].plot(df['speechiness'].value_counts().sort_index())
axes[3][0].set_title('speechiness')
axes[3][1].plot(df['popularity'].value_counts().sort_index())
axes[3][1].set_title('popularity')
fig,axes = plt.subplots(nrows=4, ncols=3, figsize=(14,10))
plt.tight_layout()
sns.distplot(df['beats_per_minute'], color='r', ax=axes[0][0])
sns.distplot(df['energy'], color='b', ax=axes[0][1])
sns.distplot(df['danceability'], color='r', ax=axes[0][2])
sns.distplot(df['liveness'], color='b', ax=axes[1][0])
sns.distplot(df['valence'], color='r', ax=axes[1][1])
sns.distplot(df['length'], color='b', ax=axes[1][2])
sns.distplot(df['loudness'], color='r', ax=axes[2][0])
sns.distplot(df['speechiness'], color='b', ax=axes[2][1])
sns.distplot(df['length'], color='r', ax=axes[2][2]) 
sns.distplot(df['popularity'], color='b', ax=axes[3][1])
fig,axes = plt.subplots(nrows=4, ncols=3, figsize=(14,10))
plt.tight_layout()
sns.boxplot(df['beats_per_minute'], color='r', ax=axes[0][0])
sns.boxplot(df['energy'], color='b', ax=axes[0][1])
sns.boxplot(df['danceability'], color='r', ax=axes[0][2])
sns.boxplot(df['liveness'], color='b', ax=axes[1][0])
sns.boxplot(df['valence'], color='r', ax=axes[1][1])
sns.boxplot(df['length'], color='b', ax=axes[1][2])
sns.boxplot(df['loudness'], color='r', ax=axes[2][0])
sns.boxplot(df['speechiness'], color='b', ax=axes[2][1])
sns.boxplot(df['length'], color='r', ax=axes[2][2]) 
sns.boxplot(df['popularity'], color='b', ax=axes[3][1])
df.beats_per_minute.describe()
def grouping(x):
    if x<100:
        return '<100'
    elif x<=150:
        return '101-150'
    elif x<=200:
        return '151-200'
    else:
        return '>200'

groupes = df.beats_per_minute.apply(grouping)
values= groupes.value_counts()
labels= values.index
fig = px.pie(values = values,names= labels)
fig.update_layout(title = 'bpm_distribution')
fig.show()
df[df['beats_per_minute']>200]
#fig = px.scatter(df[df['beats_per_minute']>200], y="beats_per_minute", x="popularity", hover_name='song', color='beats_per_minute', size='acousticness')
#fig.show()
fig=px.violin(df, y='danceability', color='year', points='all', hover_name='song', hover_data=['artist'])
fig.show()
fig=px.violin(df, y = 'popularity', points ='all',color='year',hover_name='song',hover_data=['artist'])
fig.show()
fig=px.violin(df, y = 'energy', points ='all',color='year',hover_name='song',hover_data=['artist'])
fig.show()
fig = px.scatter(df,x='danceability', y='energy',color='energy',hover_name='song',hover_data=['artist','year'])
fig.show()
fig = px.scatter(df,x='popularity', y='length',color='length',hover_name='song',hover_data=['artist','year'])
fig.show()
fig = px.scatter(df,x='popularity', y='speechiness',color='speechiness',hover_name='song',hover_data=['artist','year'])
fig.show()
fig = px.scatter(df.query('year==2019'), y='popularity', x='artist', hover_name='song', color='popularity' )
fig.show()
df[df['genre'].str.contains('pop')]
fig = px.scatter(df[df['genre'].str.contains('pop')], x='artist', y='popularity', hover_name='song', hover_data=['year','artist'] )
fig.show()
df['artist'].value_counts().head(10)
kp = df[df['artist']=='Katy Perry']
kp
df[df['artist']=='Katy Perry']['year'].value_counts()
fig=px.scatter(df[df['artist']=='Katy Perry'], y='popularity', x= 'year', hover_name='song', hover_data=['artist','year'])
fig.show()
df.sort_values(by='acousticness', ascending = False).head(10)[['song','artist','genre','year','acousticness']]
df.sort_values(by='liveness',ascending=False).head(10)[['song','artist','genre','year','liveness']]
corrr = df.corr()
fig=plt.figure(figsize=(10,8))
sns.heatmap(corrr, annot=True,cmap='GnBu_r', center=1)
