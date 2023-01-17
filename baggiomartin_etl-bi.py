import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import seaborn as sns

from pathlib import Path

import IPython

import IPython.display

from sklearn import datasets, linear_model

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE



%matplotlib inline
DATA = Path('../input')

song_atrib = DATA/'spotifyclassification/data.csv'

song_world = DATA/'spotifys-worldwide-daily-song-ranking/data.csv'

song_tracks = DATA/'ultimate-spotify-tracks-db/SpotifyFeatures.csv'

df_atrib = pd.read_csv(song_atrib)

df_world = pd.read_csv(song_world)

df_tracks = pd.read_csv(song_tracks)
df=pd.DataFrame()

df = pd.merge(df_world, df_tracks, how='left', left_on=['Track Name','Artist'], 

              right_on = ['track_name','artist_name'])

df = df.dropna()

df = df.drop(['track_name','artist_name', 'URL'], axis=1)

df = df[df['Date']<'2018-01-01']

df = df.drop_duplicates().reset_index(drop=True)
condition= df['Position']<=20

df.where(cond=condition,inplace=True)

condition= df.Region!='global'

df.where(cond=condition,inplace=True)
df = df.dropna().reset_index(drop=True)
df = df.drop_duplicates(['track_id', 'Region', 'Date']).reset_index(drop=True)
df['Region']=df['Region'].str.upper()
df['Date'] = pd.to_datetime(df['Date'])

df['Date2'] = df['Date'].dt.strftime('%d%m%Y')
df.to_csv('world_df.csv', index=False)
df[['Region', 'Streams']].groupby('Region').mean()
df[df['Region']=='US']['Track Name'].nunique()
df[df['Region']=='AR']['Track Name'].nunique()
df[df['Track Name']=='La Pegajosa']
df.groupby('Artist')['Streams'].sum()*0.006