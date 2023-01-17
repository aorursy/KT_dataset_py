#I am going to use SpotiPy library to work with Spotify API
!pip install spotipy


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import spotipy
import spotipy.util as util
import json

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# secrets are used to store Spotify API client ID and secret
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
SPOTIFY_CLIENT_ID = user_secrets.get_secret("spotify_client_id")
SPOTIFY_CLIENT_SECRET = user_secrets.get_secret("spotify_client_secret")

# constuct API handler
from spotipy.oauth2 import SpotifyClientCredentials
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
#These are the tracks (unique list)
hot100 = pd.read_csv("/kaggle/input/data-on-songs-from-billboard-19992019/BillboardFromLast20/billboardHot100_1999-2019.csv").drop(columns=['Unnamed: 0','Weekly.rank','Peak.position','Weeks.on.chart','Week','Genre','Writing.Credits','Lyrics','Features']).drop_duplicates()
hot100.head()
len(hot100)
# These are the artists and a number of songs by 10 of them
hot100_artists = hot100['Artists'].drop_duplicates()
len(hot100_artists)
len(hot100[hot100['Artists'].isin(hot100_artists.head(20))])

# Limit number of artists to survey
ARTIST_LIMIT=300

#this function takes care of one track. It accepts track record and returns semi-parsed information from Spotify API
def searchSp (track):
    #Search for the track
    raw_search=sp.search(q='artist:"'+track.Artists+'" track:"'+track.Name+'"', type='track')
    if len(raw_search['tracks']['items'])>0:
        #track is found, let's take the first search result
        p1=raw_search['tracks']['items'][0]
        #let's extract album information
        album=pd.json_normalize(p1['album'])[['id', 'name']]
        
        #and artist information
        artist=pd.json_normalize(p1['artists'][0])[['id','name']]
        
        #and track info
        track_info = pd.json_normalize(p1)[['id','name','popularity']]
        
        #now let's get audio features
        a1 = sp.audio_features(track_info['id'])
        audio_features=pd.json_normalize(a1[0]).drop(columns=['type', 'id','uri','track_href', 'analysis_url', 'duration_ms'])
        
        #put everything together, into single line dataframe
        result=pd.concat([track_info.add_prefix('track.'), audio_features.add_prefix('audio.'), album.add_prefix('album.'), artist.add_prefix('artist.')], axis=1)             
    else:    
        result = pd.DataFrame( columns=['track.id','track.name','track.popularity','audio.danceability','audio.energy','audio.key','audio.loudness','audio.mode','audio.speechiness','audio.acousticness','audio.instrumentalness','audio.liveness','audio.valence','audio.tempo','audio.time_signature','album.id','album.name' , 'artist.id', 'artist.name'])
    return result#.to_dict(orient='list')


sp_h100 = pd.DataFrame( columns=['track.id','track.name','track.popularity','audio.danceability','audio.energy','audio.key','audio.loudness','audio.mode','audio.speechiness','audio.acousticness','audio.instrumentalness','audio.liveness','audio.valence','audio.tempo','audio.time_signature','album.id','album.name' , 'artist.id', 'artist.name'])
#iterate through hot100 and pull track data from Spotify for each track by selected artist
for tr in hot100[hot100['Artists'].isin(hot100_artists.head(ARTIST_LIMIT))].itertuples():
    sp_h100=sp_h100.append(searchSp(tr))
#let's save our hot track list as csv
sp_h100.to_csv('Spotify_data_for_hot_100_select.csv')
sp_h100.head(5)
#let's retrive hot track list from csv
sp_h100=pd.read_csv('/kaggle/working/Spotify_data_for_hot_100_select.csv').drop(columns=['Unnamed: 0'])
print("Hot 100 before de-duplication", len(sp_h100))
sp_h100=sp_h100.drop_duplicates(subset='track.id')
print("Hot 100 after de-duplication", len(sp_h100))
sp_h100.head()
#this function retrieves all tracks by given artist

def tracks_by_artist_Sp (a):
    #Search for the track
    raw_search=sp.artist_top_tracks(a[1])
    if len(raw_search['tracks'])>0:
        raw_tracks = pd.json_normalize(pd.json_normalize(raw_search).explode('tracks')['tracks'])
    
         #now let's get audio features
        a1 = sp.audio_features(raw_tracks['id'].tolist())
        audio_features = pd.json_normalize(a1).drop(columns=['type', 'uri', 'track_href', 'analysis_url', 'duration_ms']).add_prefix("audio.")
        result=raw_tracks[['id', 'name', 'popularity']].add_prefix('track.').merge(audio_features, left_on='track.id', right_on='audio.id').drop(columns='audio.id') 
        result=pd.concat([result, raw_tracks[['album.id', 'album.name']]],axis=1)
        result['artist.id']=a[1]
        result['artist.name']=a[2]
    else:
        result = pd.DataFrame( columns=['track.id','track.name','track.popularity','audio.danceability','audio.energy','audio.key','audio.loudness','audio.mode','audio.speechiness','audio.acousticness','audio.instrumentalness','audio.liveness','audio.valence','audio.tempo','audio.time_signature','album.id','album.name' , 'artist.id', 'artist.name'])
       
    return result

TRACK_LIMIT=len(sp_h100)
sp_a20 = pd.DataFrame( columns=['track.id','track.name','track.popularity','audio.danceability','audio.energy','audio.key','audio.loudness','audio.mode','audio.speechiness','audio.acousticness','audio.instrumentalness','audio.liveness','audio.valence','audio.tempo','audio.time_signature','album.id','album.name' , 'artist.id', 'artist.name'])
#iterate through hot100 and pull track data from Spotify for each track
for tr in sp_h100.head(TRACK_LIMIT)[['artist.id', 'artist.name']].itertuples():
    sp_a20=sp_a20.append(tracks_by_artist_Sp(tr))
print ("Tracks before deduplication", len(sp_a20))
sp_a20 = sp_a20.drop_duplicates(subset='track.id')
print ("Tracks after deduplication", len(sp_a20))
sp_a20.head(5)
sp_a20 = sp_a20[~sp_a20['track.id'].isin(sp_h100['track.id'])]
print ("Popular tracks after exclusion of hot", len(sp_a20))
sp_a20.to_csv('Spotify_data_for_popular_select.csv')
#let's retrive popular track list and hot 100 list from csv
sp_h100=pd.read_csv('/kaggle/working/Spotify_data_for_hot_100_select.csv').drop(columns=['Unnamed: 0'])
sp_h100=sp_h100.drop_duplicates(subset='track.id')
sp_a20=pd.read_csv('/kaggle/working/Spotify_data_for_popular_select.csv').drop(columns=['Unnamed: 0'])
#let's summon the forces of matplot
import matplotlib.pyplot as plt

#for visualization we are going to use several functions to decode audio analysis
def audio_key(key):
    all_keys=['C', 'C#', 'D','D#','E','F','F#','G','G#','A','A#','B']
    return all_keys[key]
def audio_meter(sig):
    all_sigs=['Unknown','1/1','1/2','3/4','4/4','5/8']
    if sig>len(all_sigs)-1: return 'Unknown'
    else: return all_sigs[sig]

plt.style.use('seaborn-poster')
hh = pd.DataFrame()
hh['Hits']=sp_h100['audio.key'].apply(audio_key).value_counts(normalize=True)
hh['Popular Songs']=sp_a20['audio.key'].apply(audio_key).value_counts(normalize=True)
hh.plot(kind='bar', title='Musical Key of Tracks in Hot 100 and Top 10 by Artist', legend=['Hits', 'Popular Songs'])
#plt.scatter(final_dataset['audio.key'],final_dataset['audio.tempo'], alpha=0.5, c='Purple') 
#lets' summarize other audio parameters
sp_h100.iloc[:,[3,4,8,9,10,11,12]].plot.box(fontsize=8)
sp_a20.iloc[:,[3,4,8,9,10,11,12]].plot.box(fontsize=8)
sp_a20['audio.tempo'].value_counts(bins=200, normalize=True)

tt = pd.DataFrame()
tt['Hits']=sp_h100['audio.tempo']
tt['Popular Songs']=sp_a20['audio.tempo']
tt.plot.hist(bins=200, alpha=0.5, density=1)