import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, timedelta

import os

print(os.listdir("../input"))
#Streaming History

df = pd.read_json('../input/spotify-vivi/StreamingHistory.json', encoding='utf-8')

df['endTime'] = pd.to_datetime(df['endTime'])

df['date'] = df['endTime'].dt.date

df['month'] = df['endTime'].dt.month

df['month'] = df['endTime'].values.astype('datetime64[M]')

df['weekEnding'] = df['date'].apply(lambda x: x - timedelta(days = x.weekday()) + timedelta(days = 6))

df['hoursPlayed'] = df['msPlayed'].apply(lambda x: x / 3600000)
df.info()
df.head()
print('Min date: {}'.format(df.endTime.min()))

print('Max date: {}'.format(df.endTime.max()))
df['qualify'] = df['msPlayed'].apply(lambda x: 1 if x >= 30_000 else 0)
GROUP_COL = 'month'



#Prep data

month_volume = df.groupby([GROUP_COL, 'qualify'], as_index = False).agg({'hoursPlayed':'sum'})

month_volume = month_volume.pivot(index = 'month', columns = 'qualify', values = 'hoursPlayed').sort_values(by = 'month', ascending = False)



month_count = df.groupby([GROUP_COL, 'qualify'], as_index = False).agg({'hoursPlayed':'count'})

month_count = month_count.pivot(index = 'month', columns = 'qualify', values = 'hoursPlayed').sort_values(by = 'month', ascending = False)



col_labels = month_volume.index



fig, ax = plt.subplots(1, 2, figsize=(20,8))

g = month_volume.plot(kind = 'barh', stacked = True, ax = ax[0])

g2 = month_count.plot(kind = 'barh', stacked = True, ax = ax[1])



# # Modify y-axis lables for dates

g.set_yticklabels([m.strftime('%b-%y') for m in col_labels])

g2.set_yticklabels([m.strftime('%b-%y') for m in col_labels])



g.set_title('Monthly Spotify Play Hours', fontsize = 24)

g2.set_title('Monthly Spotify Play Count', fontsize = 24)



ax[0].set_xlabel('Hours Played', fontsize = 16)

ax[0].set_ylabel('Month', fontsize = 16)

plt.show()
pie_chart_count = df.groupby(['qualify'], as_index = False).agg({'hoursPlayed':'count'}).rename(columns={'hoursPlayed':'playCount'}).set_index('qualify')

pie_chart_count.index = pie_chart_count.index.map({1:'Qualifies',0:'Too Short'})



pie_chart_volume = df.groupby(['qualify'], as_index = False).agg({'hoursPlayed':'sum'}).set_index('qualify')

pie_chart_volume.index = pie_chart_volume.index.map({1:'Qualifies',0:'Too Short'})



fig, ax = plt.subplots(1,2,figsize=(16,8))

pie_chart_count.plot(kind = 'pie', y = 'playCount', ax = ax[1], table = True)

pie_chart_volume.plot(kind = 'pie', y = 'hoursPlayed', ax = ax[0], table = True)

plt.show()
GROUP_COL = 'month'



#Prep data

month_volume = df.groupby([GROUP_COL], as_index = False).agg({'hoursPlayed':'sum'})

col_labels = month_volume[GROUP_COL]



fig, ax = plt.subplots(figsize=(10,8))

g = sns.barplot(x = 'hoursPlayed', y = GROUP_COL, orient = 'h', data = month_volume, ax = ax, color = 'lightblue')



# Modify y-axis lables for dates

g.set_yticklabels([m.strftime('%b-%y') for m in col_labels])

g.set_title('Monthly Spotify Play Hours', fontsize = 24)

plt.xlabel('Hours Played', fontsize = 16)

plt.ylabel('Month', fontsize = 16)

plt.show()
#Prep data

artist_volume = df.groupby(['artistName'], as_index = False).agg({'hoursPlayed':'sum'}).sort_values(by='hoursPlayed', ascending = False).head(25)

song_volume = df.groupby(['trackName'], as_index = False).agg({'hoursPlayed':'sum'}).sort_values(by='hoursPlayed', ascending = False).head(25)





fig, ax = plt.subplots(2,1, figsize=(10,20))

g1 = sns.barplot(x = 'hoursPlayed', y = 'artistName', orient = 'h', data = artist_volume, ax = ax[0], color = 'lightblue')

g2 = sns.barplot(x = 'hoursPlayed', y = 'trackName', orient = 'h', data = song_volume, ax = ax[1], color = 'lightblue')





g1.set_title('Monthly Spotify Play Hours', fontsize = 24)

ax[0].set_xlabel('Hours Played', fontsize = 16)

ax[0].set_ylabel('Artist Name', fontsize = 16)



ax[1].set_xlabel('Hours Played', fontsize = 16)

ax[1].set_ylabel('Track Name', fontsize = 16)



plt.show()
GROUP_COL = 'month'



#Prep data

month_volume = df.groupby([GROUP_COL], as_index = False).agg({'hoursPlayed':'count'})

col_labels = month_volume[GROUP_COL]



fig, ax = plt.subplots(figsize=(10,8))

g = sns.barplot(x = 'hoursPlayed', y = GROUP_COL, orient = 'h', data = month_volume, ax = ax, color = 'lightblue')



# Modify y-axis lables for dates

g.set_yticklabels([m.strftime('%b-%y') for m in col_labels])

g.set_title('Monthly Spotify Play Count', fontsize = 24)

plt.xlabel('Count of Plays', fontsize = 16)

plt.ylabel('Month', fontsize = 16)

plt.show()
#Prep data

artist_volume = df.groupby(['artistName'], as_index = False).agg({'hoursPlayed':'count'}).sort_values(by='hoursPlayed', ascending = False).head(25)

song_volume = df.groupby(['trackName'], as_index = False).agg({'hoursPlayed':'count'}).sort_values(by='hoursPlayed', ascending = False).head(25)





fig, ax = plt.subplots(2,1, figsize=(10,20))

g1 = sns.barplot(x = 'hoursPlayed', y = 'artistName', orient = 'h', data = artist_volume, ax = ax[0], color = 'lightblue')

g2 = sns.barplot(x = 'hoursPlayed', y = 'trackName', orient = 'h', data = song_volume, ax = ax[1], color = 'lightblue')





g1.set_title('Monthly Spotify Play Count', fontsize = 24)

ax[0].set_xlabel('Count of Plays', fontsize = 16)

ax[0].set_ylabel('Artist Name', fontsize = 16)



ax[1].set_xlabel('Count of Plays', fontsize = 16)

ax[1].set_ylabel('Track Name', fontsize = 16)



plt.show()