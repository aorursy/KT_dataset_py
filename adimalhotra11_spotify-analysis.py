# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv',index_col=0)
data.head()
data.columns
data.shape
data.info
data.sort_values(by='release_date',ascending=False)
print("MAXIMUM DURATION SONG - " , np.max(data.duration_ms)/(1000*60), "minutes")

print("MINIMUM DURATION SONG - " , np.min(data.duration_ms)/(1000*60) , "minutes")

print("Average song duration : ", np.mean(data.duration_ms)/(1000*60) , "minutes")
data.groupby('loudness').agg({'popularity':'mean'}).sort_values(by='popularity',ascending=False).hist()
plt.figure(figsize=(16, 8))

sns.distplot(data.popularity,bins=10)
gr_yr = data.groupby('year').agg({'duration_ms' : 'mean'})

print(gr_yr.tail(20))
plt.figure(figsize=(14,5))

sns.barplot(x = gr_yr.index[-10:] , y=gr_yr.duration_ms[-10:] , data = gr_dur)
gr_yr = data.groupby('year').agg({'popularity' : 'sum'})

print(gr_yr)
plt.figure(figsize=(14,4))

sns.barplot(x = gr_yr.index[-10:] , y=gr_yr.popularity[-10:] , data = gr_yr)
print('50 Most Popular Songs on Spotify!')

data[['name','artists','release_date','popularity']].sort_values(by='popularity',ascending=False).head(50)
data[data["popularity"] > 90]["name"]

most_pop_artists = data.groupby('artists').count().sort_values(by='popularity',ascending=False)
print('20 most Popular Artists on Spotify : \n')

for i in most_pop_artists.head(20).index:

    print(i,end='\n')

    

data.corr()
corrmat = data.corr()

top_corr_feat = corrmat.index

plt.figure(figsize=(12,12))

h = sns.heatmap(data[top_corr_feat].corr(),annot=True,cmap="RdYlGn")

plt.figure(figsize=(18,7))

plt.xlabel('year')

plt.ylabel('popularity')

plt.xticks(rotation=45)

plt.title('Popularity of Songs by Year')

plt.gca().margins(x=0)

plt.gcf().canvas.draw()

sns.barplot(x='year',y='popularity',data=data.sort_values(by='year',ascending=False))

plt.show()
plt.figure(figsize = (12,6))

sns.set(style='whitegrid')

x = data.groupby('year')['id'].count()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('song count')

plt.figure(figsize = (12,6))

sns.set(style='whitegrid')

x = data.groupby('year')['danceability'].mean()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('danceability of songs')
plt.figure(figsize = (12,6))

x = data.groupby('year')['explicit'].mean().sort_values(ascending=False)

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('explicit songs count')
plt.figure(figsize = (12,6))

x = data.groupby('year')['loudness'].mean().sort_values(ascending=False)

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('loudness')
plt.figure(figsize = (12,6))

x = data.groupby('year')['tempo'].mean()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('tempo')
plt.figure(figsize = (12,6))

x = data.groupby('year')['energy'].mean()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('energy')
plt.figure(figsize = (12,6))

x = data.groupby('year')['instrumentalness'].mean()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('instrumentalness')
plt.figure(figsize = (12,6))

x = data.groupby('year')['speechiness'].mean()

ax = sns.lineplot(x.index , x)

ax.set_xlabel('year')

ax.set_ylabel('speechiness')