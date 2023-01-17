# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install 'spotipy' --upgrade

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pyplot as plt

from  matplotlib.pyplot import plot 

import json

import requests

import spotipy

from spotipy.oauth2 import SpotifyClientCredentials

import spotipy.util as util



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

cid = "23d2340cf7784b1298db98134d305025"

secret = "2029f9a734c54ee09ab5ecc1c5fe61c9"

redirect_uri = "http://localhost:8888/callback"

scopes = "user-read-private user-read-email user-read-playback-state user-read-currently-playing user-library-read"

username = "minky961203@gmail.com"
token = util.prompt_for_user_token(username=username,

                           scope=scopes,

                           client_id=cid,

                           client_secret=secret,

                           redirect_uri=redirect_uri)

sp = spotipy.Spotify(auth=token)
results = sp.current_user_saved_tracks(limit=50)

df = pd.DataFrame(results)

for item_info in df['items']:

    for keys in item_info:

        print(keys)   # print all the high level parameters in this dataset
results # printing results here to see structure of dataset
columns = ['date_added','artist_name', 'track_id', 'track_title', 'popularity']

cleaned = pd.DataFrame(columns=columns) # create a new dataframe with only the desired parameters

artist_names = [] # create empty lists for each parameter

artist_id = []

track_id = []

track_title = []

popularity = []

date_added = []



for item in df['items']: # from each item in the dataset

    date = pd.Timestamp(item['added_at'])

    track_info = item['track'] # store everything under the 'track' key

    for data in track_info:

        artist_info = track_info['artists'] # extract artist info from track info 

        artist_name = artist_info[0]['name'] # in the first item of artist info list, get all values with the key 'name', i.e. the artist names

        artist_ids = artist_info[0]['id'] # repeat for artist ids

        track_ids = track_info['id'] # repeat for track ids

        name = track_info['name'] # repeat for track titles

        p_score = track_info['popularity'] # repeat for popularity

    artist_names.append(artist_name)

    artist_id.append(artist_ids)

    track_id.append(track_ids)

    track_title.append(name)

    popularity.append(p_score)

    date_added.append(date) # append all extracted info to dataframe for each item info bundle in the dataset



cleaned['artist_name'] = artist_names

cleaned['artist_id'] = artist_id

cleaned['track_id'] = track_id

cleaned['track_title'] = track_title

cleaned['popularity'] = popularity

cleaned['date_added'] = date_added



cleaned
audio_features = pd.DataFrame(sp.audio_features(track_id)) # create dataframe of audio features from track id list from dataset above

energy_level = audio_features['energy']

valence = audio_features['valence']

audio_features # these are the audio features for each track from dataset above
descriptive_stats = pd.DataFrame(audio_features.describe())

descriptive_stats # see summary of descriptive stats of audio features
boxplot = audio_features.boxplot(column = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence'], rot=45)

# based on this boxplot, it looks like the danceability and energy of my tracks are overall higher than the rest of audio features. 
time = pd.to_datetime(date_added, format = '%y%m%d') # convert date_added column from dataset 'cleaned' to obtain desired format 

plt.plot(time, valence) # plot to see change in valence values of played tracks over time. 

plt.xlabel('date added')

plt.ylabel('valence score of tracks')



# no significant trends but seems like I hit my lowest point around December.
plt.plot(time, energy_level) # plot to see change in energy levels of played tracks over time. 

plt.xlabel('date added')

plt.ylabel('energy level of tracks')



# no significant trends here again but I also hit my lowest point in December.
plt.scatter(audio_features['danceability'], energy_level) # here are some plots of the correlation between various features

plt.xlabel('danceability') 

plt.ylabel('energy level')
plt.scatter(audio_features['acousticness'], energy_level) # here are some plots of the correlation between various features

plt.xlabel('acousticness') 

plt.ylabel('energy level')
plt.scatter(audio_features['liveness'], energy_level) # here are some plots of the correlation between various features

plt.xlabel('liveness') 

plt.ylabel('energy level')
plt.scatter(audio_features['danceability'], audio_features['acousticness']) # here are some plots of the correlation between various features

plt.xlabel('danceability') 

plt.ylabel('acousticness')
plt.scatter(audio_features['liveness'], audio_features['acousticness']) # here are some plots of the correlation between various features

plt.xlabel('liveness') 

plt.ylabel('acousticness')
plt.scatter(valence, energy_level) # here are some plots of the correlation between various features

plt.xlabel('valence') 

plt.ylabel('energy level')
plt.scatter(popularity, energy_level) # plot to check correlation between popularity score and energy level

plt.xlabel('popularity score') 

plt.ylabel('energy level')



# it seems like there are no correlations between whether a track is popular and its energy level. 
plt.scatter(popularity, valence) # plot to check correlation between popularity score and energy level

plt.xlabel('popularity score') 

plt.ylabel('valence')
pop_energy = np.corrcoef(popularity, energy_level) # as shown by correlation matrix here, the correlation coefficient is -0.247, which suggests no relationship between two variables

pop_valence = np.corrcoef(popularity, valence) # same for valence. Does not necessarily mean that greater the valence of tracks, the more popular.



print(pop_energy)

print(pop_valence)