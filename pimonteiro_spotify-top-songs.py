import pandas as pd

import seaborn as sn

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1')

data.head()
data.info()
data.count()
data = data.rename(columns={'top genre': 'top_genre'})

data = data.drop('Unnamed: 0', axis=1)

print(data.columns)
artists = data['artist'].value_counts().reset_index().head(10)

print(artists)
plt.figure(figsize=(15,10))

sn.barplot(x='index',y='artist', data=artists)

plt.title("Number of Musics on top score by the 10 Top Artists")
plt.figure(figsize=(20,10))

for i in artists['index']:

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['artist'] == i]

    tmp.append(songs.shape[0])

  sn.lineplot(x=list(range(2010,2020)),y=tmp)

plt.legend(list(artists['index']))

plt.title("Evolution of each Top 10 Artists throught the target Years")
data['artist'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')
data[data['artist'] == 'Lady Gaga'][data['year'] == 2018]
plt.figure(figsize=(20,40))

graph = 0

for i in data['artist'].unique():

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['artist'] == i]

    tmp.append(songs.shape[0])

  graph = sn.lineplot(x=list(range(2010,2020)),y=tmp, label=i)

fig = graph.get_figure()

fig.savefig("artists.png")
genres = data['top_genre'].value_counts().reset_index().head(10)
plt.figure(figsize=(23,10))

sn.barplot(x='index',y='top_genre', data=genres)
data['top_genre'].value_counts().head(10).plot.pie(figsize=(8,8), autopct='%1.0f%%')
plt.figure(figsize=(20,10))

for i in genres['index']:

  tmp = []

  for y in range(2010,2020):

    songs = data[data['year'] == y][data['top_genre'] == i]

    tmp.append(songs.shape[0])

  sn.lineplot(x=list(range(2010,2020)),y=tmp)

plt.legend(list(genres['index']))
def mean_of(col):

  res = []

  years = list(range(2010,2020))

  for y in years:

    tmp = data[data['year'] == y][col]

    res.append(np.mean(tmp))

  return res
# Relation of the music attributes with each year's mode

plt.figure(figsize=(15,5))

years = list(range(2010,2020))

res = mean_of('bpm')

sn.lineplot(x=years,y=res)

res = mean_of('nrgy')

sn.lineplot(x=years,y=res)

res = mean_of('dnce')

sn.lineplot(x=years,y=res)

res = mean_of('dB')

sn.lineplot(x=years,y=res)

res = mean_of('live')

sn.lineplot(x=years,y=res)

res = mean_of('val')

sn.lineplot(x=years,y=res)

res = mean_of('dur')

sn.lineplot(x=years,y=res)

res = mean_of('acous')

sn.lineplot(x=years,y=res)

res = mean_of('spch')

sn.lineplot(x=years,y=res)

plt.legend(['bpm','nrgy','dnce','dB','live','val','dur','acous','spch'])

plt.title('(mean) Music Attributes per Year')
plt.figure(figsize=(15,5))

for y in range(2010,2020):

  tmp = data[data['year'] == y]['val']

  tmp.plot.line()
plt.figure(figsize=(15,5))

for y in range(2010,2020):

  tmp = data[data['year'] == y]['nrgy']

  tmp.plot.line()
plt.figure(figsize=(15,5))

for y in range(2010,2020):

  tmp = data[data['year'] == y]['live']

  tmp.plot.line()

  tmp2 = data[data['year'] == y]['pop']

  tmp2.plot.line()
print(data[data['pop'] == max(data['pop'])])

print(data[data['pop'] == min(data['pop'])])
data.sort_values(by=['live'], ascending=False).head(15)
data.sort_values(by=['pop'], ascending=False).head(15)
data.sort_values(by=['dur'], ascending=False).head(15)
data.sort_values(by=['acous'], ascending=False).head(15)