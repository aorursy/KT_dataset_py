import numpy as np

import pandas as pd

from sklearn import preprocessing
df = pd.read_csv("../input/spotifyclassification/data.csv")

print("\n  There are",len(df),"songs in the dataset\n")
features = ['acousticness', 'danceability', 'duration_ms', 'energy',

       'instrumentalness', 'key', 'liveness', 'loudness', 'mode',

       'speechiness', 'tempo', 'time_signature', 'valence']

print("\n  There are 13 song characteristics in the dataset. They are stored as the features:\n")

for x in features:

    print("  ",x)

print()
print("\n  Standardizing the song features.\n  The columns are now in a z-distributions.\n")

# Create the Scaler object

scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

# Fit your data on the scaler object

dfz = scaler.fit_transform(df[features])

dfz = pd.DataFrame(dfz, columns=features)

dfz['song_title']=df['song_title']

dfz['artist']=df['artist']
print("\n  10 Random Songs from the Dataset:\n")

random_songs = df.loc[np.random.permutation(df.index)[:10]]['song_title'].values

for song in random_songs:

    print("  ",song)

print()
title = random_songs[0]

target_row = dfz.loc[dfz['song_title'] == title].head(1)

artist = target_row.iloc[0]['artist']

print("\n  We can now find the songs most similar to",title,"by",artist,"\n  (the first random song selected)\n")
print("\n  Retrieving the song characteristics (in z-scores) for each feature of",title,"by",artist,"\n")

acousticness = target_row.iloc[0]['acousticness'] 

danceability = target_row.iloc[0]['danceability']

instrumentalness = target_row.iloc[0]['instrumentalness'] 

duration_ms = target_row.iloc[0]['duration_ms'] 

energy = target_row.iloc[0]['energy'] 

key = target_row.iloc[0]['key'] 

liveness = target_row.iloc[0]['liveness']

loudness = target_row.iloc[0]['loudness'] 

mode = target_row.iloc[0]['mode'] 

speechiness = target_row.iloc[0]['speechiness'] 

tempo = target_row.iloc[0]['tempo'] 

time_signature = target_row.iloc[0]['time_signature'] 

valence = target_row.iloc[0]['valence']
print("\n  Song Characteristics\n  -----------------------------------------")

for col in list(target_row)[::-1]:

    try:

        print(" ",col," "*(30-len(col)),round(target_row.iloc[0][col],3))

    except:

        print(" ",col," "*(30-len(col)),target_row.iloc[0][col])

print()
dfz['similarity'] = abs(dfz['acousticness'] - acousticness) + abs(dfz['danceability'] - danceability)

+ abs(dfz['duration_ms'] - duration_ms) + abs(dfz['energy'] - energy) + abs(dfz['instrumentalness'] - instrumentalness) 

+ abs(dfz['key'] - key) + abs(dfz['liveness'] - liveness) + abs(dfz['loudness'] - loudness)

+ abs(dfz['mode'] - mode) + abs(dfz['danceability'] - danceability) + abs(dfz['tempo'] - tempo)

+ abs(dfz['speechiness'] - speechiness) + abs(dfz['valence'] - valence)

print("\n  Computing the similarity score for all songs in the dataset with the song\n ",title,"by",artist,"\n")
dfz_sims = dfz[dfz.song_title != title].nsmallest(10, 'similarity')

print("\n  Top 10 Songs Most Similar to:\n ",title,"-",artist,"\n")

score = 1

for ind, row in dfz_sims.iterrows():

    print("   ",score,row['song_title'],"-",row['artist'])

    score+=1

print()