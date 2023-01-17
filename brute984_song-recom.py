# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_songs = pd.read_csv('../input/SpotifyFeatures.csv')
df_songs.info()
df_songs.head(3)
df_songs.drop('key',inplace = True, axis = 1)

df_songs.drop('time_signature', inplace = True, axis = 1)

df_songs.drop('duration_ms',inplace=True,axis=1)

df_songs.drop('mode',inplace=True,axis=1)

df_songs.drop('valence',inplace=True, axis=1)

plt.figure(figsize=(15,10))

ax = sns.countplot(x='genre',

              data=df_songs, 

             order=df_songs['genre'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right")

plt.title("Genre",fontsize=35,color="white")

plt.show()
plt.figure(figsize=(12,10))

ax = sns.countplot(x='artist_name',

              data=df_songs, 

             order=pd.value_counts(df_songs['artist_name']).iloc[:10].index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right")

plt.title("Top Artist",fontsize=35,color="white")

plt.show()
df_songs.head(1)
df_missing = df_songs.isnull().sum(axis=0)

df_missing.columns = ['column_name', 'missing_count']

df_missing.head(20)
f, ax = plt.subplots(figsize=(12, 9))

#_____________________________

# calculations of correlations

corrmat = df_songs.dropna(how='any').corr()

#________________________________________

k = 17 # number of variables for heatmap

cols = corrmat.nlargest(k, 'popularity')['popularity'].index

cm = np.corrcoef(df_songs[cols].dropna(how='any').values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, linewidth = 0.1, cmap = 'coolwarm',

                 yticklabels=cols.values, xticklabels=cols.values)

f.text(0.5, 0.93, "Correlation coefficients", ha='center', fontsize = 18, family='fantasy')

plt.show()
df_songs.head(1)
from scipy import spatial



def Similarity(songs1, songs2):

    a = df_songs.iloc[songs1]

    b = df_songs.iloc[songs2]

    

    instrumentalnessA = a['instrumentalness']

    instrumentalnessB = b['instrumentalness']

    

    instrumentalnessDist = spatial.distance.cosine(genresA, genresB)

    

    acousticnessA = a['acousticness']

    acousticnessB = b['acousticness']

    acousticnessDist = spatial.distance.cosine(wordsA, wordsB)

    

    danceabilityA = a['danceability']

    danceabilityB = b['danceability']

    danceabilityDist = spatial.distance.cosine(crewA, crewB)

    

    energyA = a['energy']

    energyB = b['energy']

    energyDist = spatial.distance.cosine(castA, castB)

    

    return instrumentalnessDist + acousticnessDist + danceabilityDist + energyDist
Similarity(3,160)
new_id=list(range(0,df_songs.shape[0]))

df_songs['new_id']=new_id

df_songs.head(2)
import operator



def Recommended(name):

    new_song=df_songs[df_songs['track_name'].str.contains(name)].iloc[0].to_frame().T

    print('Selected Song: ',new_song.track_name.values[0])

    def getNeighbors(baseSong, K):

        distances = []

    

        for index, song in df_songs.iterrows():

            if song['new_id'] != baseSong['new_id'].values[0]:

                dist = Similarity(baseSong['new_id'].values[0], song['new_id'])

                distances.append((song['new_id'], dist))

    

        distances.sort(key=operator.itemgetter(1))

        neighbors = []

    

        for x in range(K):

            neighbors.append(distances[x])

        return neighbors



    K = 10

    avgRating = 0

    neighbors = getNeighbors(new_song, K)

    

    for neighbor in neighbors:

        avgRating = avgRating+df_songs.iloc[neighbor[0]][2]  

        print( df_songs.iloc[neighbor[0]][0]+" | Genres: "+str(df_songs.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(df_songs.iloc[neighbor[0]][2]))
Recommended('Stiffelio, Act III: Ei fugge! … Lina, pensai che un angelo … Oh gioia inesprimbile')