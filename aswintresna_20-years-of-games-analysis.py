import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime 

%matplotlib inline
sns.set_palette('Set3',10)

sns.palplot(sns.color_palette())

sns.set_context('talk')
rawData = pd.read_csv('../input/ign.csv')
rawData.head(5)
releaseDate = rawData.apply(lambda x: pd.datetime.strptime

                            ('{0} {1} {2} 00:00:00'.format(

                                x['release_year'],x['release_month'],

                            x['release_day']), '%Y %m %d %H:%M:%S'),axis=1)

rawData['releaseDate'] = releaseDate
rawData[rawData.release_year == 1970]
data = rawData[rawData.release_year > 1970]

len(data)
data.score_phrase.unique()
data.groupby('score_phrase')['score'].mean().sort_values()
data.platform.unique()
plt.figure(figsize=(15,8))

data.groupby(['release_day']).size().plot(c='b')

plt.xticks(range(1,32,3))

plt.tight_layout
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10), sharex=True)

data.releaseDate.dt.weekday.plot.kde(ax=ax[0],c='b')

data.groupby(data.releaseDate.dt.weekday).size().plot(ax=ax[1],c='r')

plt.xlim(0.,6.)

plt.xticks(range(7),['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

plt.tight_layout
plt.figure(figsize=(17,8))

plt.xticks(range(1,13),['January','February','March','April',

                       'May','June','July','August','September',

                       'October','November','December'])

data.groupby(['release_month']).size().plot(c='r')
plt.figure(figsize=(17,8))

data.groupby(['release_year']).size().plot(kind='bar')
table = data.groupby('releaseDate').size()

f, ax = plt.subplots(2,1,figsize=(17,10))

table.plot(ax=ax[0],c='r')

ax[0].set_xlabel('')

table.resample('M').mean().plot(c='orange',ax=ax[1])
data.head(2)
data.platform.value_counts()[:10].plot.pie(figsize=(5,5))
f, ax = plt.subplots(2,2,figsize=(10,10))

lastGames = data[data.release_year == 2014]

lastPopular = lastGames.platform.value_counts()[lastGames.platform.value_counts()> 5]

lastPopular.plot.pie(ax=ax[0,0])

ax[0,0].set_title('2014')

ax[0,0].set_ylabel('')

lastGames = data[data.release_year == 2015]

lastPopular = lastGames.platform.value_counts()[lastGames.platform.value_counts()> 5]

lastPopular.plot.pie(ax=ax[0,1])

ax[0,1].set_title('2015')

ax[0,1].set_ylabel('')

lastGames = data[data.release_year == 2016]

lastPopular = lastGames.platform.value_counts()[lastGames.platform.value_counts()> 5]

lastPopular.plot.pie(ax=ax[1,0])

ax[1,0].set_title('2016')

ax[1,0].set_ylabel('')

lastGames = data[data.release_year <= 2000]

lastPopular = lastGames.platform.value_counts()[lastGames.platform.value_counts()> 5]

lastPopular.plot.pie(ax=ax[1,1])

ax[1,1].set_title('2000 and Older')

ax[1,1].set_ylabel('')

plt.tight_layout