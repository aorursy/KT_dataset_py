import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, timedelta

import os

print(os.listdir("../input"))
#Streaming History

df = pd.read_json('../input/StreamingHistory.json', encoding='utf-8')

df['endTime'] = pd.to_datetime(df['endTime'])

df.set_index('endTime', inplace = True)



# We choose 30 seconds as the minimum time elapsed in order to count as a play.

df['qualify'] = df['msPlayed'].apply(lambda x: 1 if x >= 30_000 else 0)
df.info()
df.head()
print('Min date: {}'.format(df.index.min()))

print('Max date: {}'.format(df.index.max()))
monthly_play_counts = df['msPlayed'].resample('MS').count()

column_names = monthly_play_counts.index



fig, ax = plt.subplots(1,1,figsize=(10,10))

g = monthly_play_counts.plot(kind = 'barh', ax = ax)



#Plot Formatting

g.set_yticklabels([m.strftime('%b-%y') for m in column_names])

ax.set_xlabel('Play Count', fontsize = 16)

ax.set_ylabel('Month', fontsize = 16)

plt.show()
monthly_hours_played = df['msPlayed'].resample('MS').sum() / 60_000 / 60

column_names = monthly_hours_played.index



fig, ax = plt.subplots(1,1,figsize=(10,10))

g = monthly_hours_played.plot(kind = 'barh', ax = ax)



#Plot Formatting

g.set_yticklabels([m.strftime('%b-%y') for m in column_names])

ax.set_xlabel('Hours Played', fontsize = 16)

ax.set_ylabel('Month', fontsize = 16)

plt.show()
monthly_mean_song_playtime = df['msPlayed'].resample('MS').mean() / 60_000

# monthly_mean_song_playtime.plot()

column_names = monthly_mean_song_playtime.index



fig, ax = plt.subplots(1,1,figsize=(10,10))

g = monthly_mean_song_playtime.plot(kind = 'line', ax = ax)



#Plot Formatting

# g.set_yticklabels([m.strftime('%b-%y') for m in column_names])

ax.set_xlabel('Month', fontsize = 16)

ax.set_ylabel('Mean Play Time (Minutes)', fontsize = 16)

plt.show()
#Prep data

artist_volume = df.groupby(['artistName'], as_index = False).agg({'msPlayed':'count'}).sort_values(by='msPlayed', ascending = False).head(25)

song_volume = df.groupby(['trackName'], as_index = False).agg({'msPlayed':'count'}).sort_values(by='msPlayed', ascending = False).head(25)





fig, ax = plt.subplots(2,1, figsize=(10,20))

g1 = sns.barplot(x = 'msPlayed', y = 'artistName', orient = 'h', data = artist_volume, ax = ax[0], color = 'lightblue')

g2 = sns.barplot(x = 'msPlayed', y = 'trackName', orient = 'h', data = song_volume, ax = ax[1], color = 'lightblue')





g1.set_title('Spotify Play Count', fontsize = 24)

ax[0].set_xlabel('Count of Plays', fontsize = 16)

ax[0].set_ylabel('Artist Name', fontsize = 16)



ax[1].set_xlabel('Count of Plays', fontsize = 16)

ax[1].set_ylabel('Track Name', fontsize = 16)



plt.show()
#Prep data

artist_time_played = df.groupby(['artistName'], as_index = False).agg({'msPlayed':'sum'}).sort_values(by='msPlayed', ascending = False)

artist_time_played['minsPlayed'] = artist_time_played['msPlayed'] / 60_000

artist_time_played = artist_time_played.head(25)



song_time_played = df.groupby(['trackName'], as_index = False).agg({'msPlayed':'sum'}).sort_values(by='msPlayed', ascending = False)

song_time_played['minsPlayed'] = song_time_played['msPlayed'] / 60_000

song_time_played = song_time_played.head(25)



fig, ax = plt.subplots(2,1, figsize=(10,20))

g1 = sns.barplot(x = 'minsPlayed', y = 'artistName', orient = 'h', data = artist_time_played, ax = ax[0], color = 'lightblue')

g2 = sns.barplot(x = 'minsPlayed', y = 'trackName', orient = 'h', data = song_time_played, ax = ax[1], color = 'lightblue')





g1.set_title('Spotify Play Minutes', fontsize = 24)

ax[0].set_xlabel('Total Minutes Played', fontsize = 16)

ax[0].set_ylabel('Artist Name', fontsize = 16)



ax[1].set_xlabel('Total Minutes Played', fontsize = 16)

ax[1].set_ylabel('Track Name', fontsize = 16)



plt.show()
#Prep data

artist_avg_time_played = df.groupby(['artistName'], as_index = False).agg({'msPlayed':'mean'}).sort_values(by='msPlayed', ascending = False)

artist_avg_time_played['minsPlayed'] = artist_avg_time_played['msPlayed'] / 60_000

artist_avg_time_played = artist_avg_time_played.head(25)



song_avg_time_played = df.groupby(['trackName'], as_index = False).agg({'msPlayed':'mean'}).sort_values(by='msPlayed', ascending = False)

song_avg_time_played['minsPlayed'] = song_avg_time_played['msPlayed'] / 60_000

song_avg_time_played = song_avg_time_played.head(25)



fig, ax = plt.subplots(2,1, figsize=(10,20))

g1 = sns.barplot(x = 'minsPlayed', y = 'artistName', orient = 'h', data = artist_avg_time_played, ax = ax[0], color = 'lightblue')

g2 = sns.barplot(x = 'minsPlayed', y = 'trackName', orient = 'h', data = song_avg_time_played, ax = ax[1], color = 'lightblue')





g1.set_title('Spotify Avg Play Minutes', fontsize = 24)

ax[0].set_xlabel('Avg Minutes Played', fontsize = 16)

ax[0].set_ylabel('Artist Name', fontsize = 16)



ax[1].set_xlabel('Avg Minutes Played', fontsize = 16)

ax[1].set_ylabel('Track Name', fontsize = 16)

ax[1].set_yticklabels([lab[0:60] for lab in song_avg_time_played['trackName']])



plt.show()
song_avg_time_played['trackName'].apply(lambda x: len(x)).plot(kind='hist')