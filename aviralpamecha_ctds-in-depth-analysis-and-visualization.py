import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
episodes = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')
plt.figure(figsize=(10,10)) 

sns.heatmap(episodes.isnull(), yticklabels=False, cbar=False, cmap='viridis')
len(episodes)
episodes.drop(index=[46,47,48,49,50,51,52,53,54], inplace=True)
episodes.head()
episodes.isnull().sum()
def impute_username(cols):

    name = cols[0]

    if pd.isnull(name):

        

        return 'unknown'

    else:

        return name
def impute_num(cols):

    num = cols[0]

    if pd.isnull(num):

        return 0

    else:

        return num
episodes['heroes_kaggle_username'] = episodes[['heroes_kaggle_username']].apply(impute_username, axis=1)
episodes['heroes_twitter_handle'] = episodes[['heroes_twitter_handle']].apply(impute_username, axis=1)
episodes['heroes'] = episodes[['heroes']].apply(impute_username, axis=1)
episodes['heroes_gender'] = episodes[['heroes_gender']].apply(impute_username, axis=1)
episodes['heroes_location'] = episodes[['heroes_location']].apply(impute_username, axis=1)
episodes['heroes_nationality'] = episodes[['heroes_nationality']].apply(impute_username, axis=1)
episodes['anchor_url'] = episodes[['anchor_url']].apply(impute_username, axis=1)
episodes['anchor_thumbnail_type'] = episodes[['anchor_thumbnail_type']].apply(impute_username, axis=1)
episodes['anchor_plays'] = episodes[['anchor_plays']].apply(impute_num, axis=1)

episodes['spotify_starts'] = episodes[['spotify_starts']].apply(impute_num, axis=1)

episodes['spotify_streams'] = episodes[['spotify_streams']].apply(impute_num, axis=1)

episodes['spotify_listeners'] = episodes[['spotify_listeners']].apply(impute_num, axis=1)

episodes['apple_listeners'] = episodes[['apple_listeners']].apply(impute_num, axis=1)

episodes['apple_avg_listen_duration'] = episodes[['apple_avg_listen_duration']].apply(impute_num, axis=1)

episodes['apple_avg_listen_duration'] = episodes[['apple_avg_listen_duration']].apply(impute_num, axis=1)

episodes['apple_listened_hours'] = episodes[['apple_listened_hours']].apply(impute_num, axis=1)

episodes.isnull().sum()
episodes['heroes_location'].unique()
plt.figure(figsize=(25,20)) 

sns.countplot(x='heroes_location',  data=episodes)
plt.figure(figsize=(25,20)) 

sns.countplot(x='heroes_nationality',  data=episodes)
plt.figure(figsize=(15,10)) 

sns.countplot(x='flavour_of_tea',  data=episodes)
episodes['apple_avg_listen_duration']
sns.countplot(x='category',  data=episodes)
plt.figure(figsize=(25,20)) 



sns.countplot(x='heroes_nationality',data=episodes,hue='heroes_gender')
sns.scatterplot(x='episode_duration',y='youtube_impression_views',data=episodes)
sns.scatterplot(x='episode_duration',y='youtube_avg_watch_duration',data=episodes,)
sns.scatterplot(x='youtube_avg_watch_duration',y='youtube_likes',data=episodes,)
plt.figure(figsize=(20,20))

sns.heatmap(episodes.corr(),annot=True,cmap='viridis')

episodes.corr()
episodes.corr()['episode_duration'].sort_values()
episodes.corr()['youtube_avg_watch_duration'].sort_values()
plt.figure(figsize=(10,7))

sns.distplot(episodes['episode_duration'], color='Green',  bins = 30 )
plt.figure(figsize=(10,7))

sns.distplot(episodes['apple_avg_listen_duration'], color='Blue')