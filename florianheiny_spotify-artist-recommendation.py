# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import ast

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv',index_col=0)

df_genre = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv')

df_artist = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_artist.csv')

df_year = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_year.csv')

df_genre2 = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

df_super_genres = pd.read_json('/kaggle/input/spotify-dataset-19212020-160k-tracks/super_genres.json')
def get_artists_collaborated_with(artist):

    flatten = lambda l: [item for sublist in l for item in sublist]

    artists, counts = np.unique([i for i in flatten([ast.literal_eval(x) for x in df[df['artists'].str.contains(artist)]['artists']]) if i != artist], return_counts=True)

    if len(artists) == 0:

        return []

    counts = counts / np.max(counts)

    indices = np.argsort(counts)

    return list(zip(artists[indices[::-1]], counts[indices[::-1]]))
def get_artists_from_same_genre(artist):

    genres = ast.literal_eval(df_genre2[df_genre2['artists'] == artist]['genres'].to_numpy()[0])

    artists = []

    for genre in genres:

        artists += df_genre2[df_genre2['genres'].str.contains('\'' + genre + '\'')]['artists'].to_list()

    if len(artists) == 0:

        return []

    artists, counts = np.unique(artists, return_counts=True)

    counts = counts / np.max(counts)

    indices = np.argsort(counts)

    return list(zip(artists[indices[::-1]], counts[indices[::-1]]))
def get_artists_with_similar_style(artist):

    df = df_artist.copy()

    df['similarity'] = np.linalg.norm(df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'tempo', 'valence']]

                                      - df[df['artists'] == artist][['acousticness', 'danceability', 'energy', 'instrumentalness', 'speechiness', 'tempo', 'valence']].to_numpy(),

                                             axis=1)

    df = df.sort_values('similarity')

    df['similarity'] = 1- df['similarity'] / df['similarity'].max()

    return list(zip(df['artists'].to_numpy()[1:], df['similarity'].to_numpy()[1:]))
def get_similar_artists(artist):

    if len(df_artist[df_artist['artists'] == artist]) == 0:

        raise NameError('Artist not found in data.')

    list_collab = get_artists_collaborated_with(artist)

    list_genre  = get_artists_from_same_genre(artist)

    list_style  = get_artists_with_similar_style(artist)

    

    artists = []

    artists += [x[0] for x in list_collab]

    artists += [x[0] for x in list_genre]

    artists += [x[0] for x in list_style]

    

    artists_unique = np.unique(artists)

    artists_dict = {a: 0 for a in artists_unique}

    

    for x in list_collab:

        artists_dict[x[0]] = artists_dict[x[0]] + x[1]

    for x in list_genre:

        artists_dict[x[0]] = artists_dict[x[0]] + x[1]

    for x in list_style:

        artists_dict[x[0]] = artists_dict[x[0]] + x[1]

        

    return {k: v for k, v in sorted(artists_dict.items(), key=lambda item: item[1], reverse=True)}
def get_similar_artists_multiple(artists, num=10):

    dict_similar = {}

    for artist, weight in artists.items():

        dict_similar[artist] = get_similar_artists(artist)

    artists_all = []

    for artist, similar_artists in dict_similar.items():

        artists_all.append(list(similar_artists.keys()))

    artists_unique = np.unique(artists_all).tolist()

    artists_dict = {artist: 0 for artist in artists_unique}

    for artist, similar_artists in dict_similar.items():

        for similar_artist, score in similar_artists.items():

            artists_dict[similar_artist] += artists[artist] * score

    return list({k: v for k, v in sorted(artists_dict.items(), key=lambda item: item[1], reverse=True) if k not in artists}.keys())[0:num]
get_similar_artists_multiple({'Eminem': 10, 'Dr. Dre': 7})
get_similar_artists_multiple({'Metallica': 10, 'Slayer': 7})
get_similar_artists_multiple({'John Williams': 10, 'Alan Silvestri': 9, 'Hans Zimmer': 8})