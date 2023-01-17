# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the data, the file has got an encoding that's not UTF-8

song_data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding="ISO-8859-1")

# Read the 2018 data from another user, this will only be used for comparing popular artists from the previous years.

song_data_2018 = pd.read_csv("../input/top-spotify-tracks-of-2018/top2018.csv")



# There was an Unnamed column which was basically the index, which I'll be dropping.

song_data.drop("Unnamed: 0",axis=1,inplace=True)

# Format the columns into a more accessible format. Strip the dots and replace with underscores.

song_data.columns = [x.lower().replace("."," ").strip().replace("  "," ").replace(" ","_") for x in song_data.columns]

# Sort the data on the Popularity metric.

song_data.sort_values("popularity",inplace=True, ascending=False)

# Generate the new index after sorting.

song_data.reset_index(drop=True,inplace=True)

# Generate an additional column that's basically a combination of the Artist and the Track.

song_data["artist_track"] = song_data.apply(lambda x: "{0}, {1}".format(x["artist_name"],x["track_name"]),axis=1)

# A little preview of the data.

song_data.head()
song_data["parent_genre"] = song_data.genre.apply(lambda x: 

{'canadian pop':"Pop",

 'reggaeton flow':"Reggae", 

 'dance pop':"Pop",

 'pop':"Pop",

 'dfw rap':"Hip Hop",

 'trap music':"Hip Hop",

 'country rap':"Country",

 'electropop':"Electronic",

 'reggaeton':"Reggae",

 'panamanian pop':"Pop",

 'canadian hip hop':"Hip Hop",

 'latin':"Pop",

 'escape room':"Escape Room",

 'pop house':"Pop",

 'australian pop':"Pop",

 'edm':"Electronic",

 'atl hip hop':"Pop",

 'big room':"Electronic",

 'boy band':"Pop",

 'r&b en espanol':"R&B",

 'brostep':"Electronic"}[x]

)
# Canvas with two plots

fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,16));



# Sort Data by Popularity, and then plot the 15 most popular tracks. The X and Y are given that way but the barh type moves it the other way round

song_data.sort_values("popularity").head(15).plot(x="artist_track",y="popularity",kind="barh",ax=ax[0],title="Top 15 Popular Tracks in 2019");



# Get the number of tracks by artist for 2019, and compare it with 2018.

pd.concat([song_data.artist_name.value_counts().rename("2019"),song_data_2018.artists.value_counts().rename("2018")],axis=1,sort=False).sort_values("2019",ascending=False).head(15)[::-1].plot(kind="barh",title="Top 15 Popular Artists",ax=ax[1]);
# Canvas with two plots

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4));



# Plot the most popular Parent Genres in 2019.

song_data.parent_genre.value_counts().plot(kind="bar",ax=ax[0],title="Popular Parent Genres");



# Plot the mot popular Genres in 2019

song_data.genre.value_counts().plot(kind="bar",ax=ax[1],title="Popular Genres");
# Just a pairplot, although I really need to consider removing one part of the graphs. It's basically the axes swapped in half the cases.

sns.pairplot(song_data[["parent_genre","beats_per_minute","energy","danceability","loudness_db","liveness","valence","length","acousticness","speechiness","popularity"]],hue="parent_genre");
fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(24,24));

plt.suptitle("Popular Tracks and their attributes");

sns.kdeplot(song_data.popularity,song_data.beats_per_minute,shade=True,ax=ax[0][0]).set_title("Popularity to Tempo");

sns.kdeplot(song_data.popularity,song_data.energy,shade=True,ax=ax[0][1]).set_title("Popularity to Energy");

sns.kdeplot(song_data.popularity,song_data.danceability,shade=True,ax=ax[0][2]).set_title("Popularity to Danceability");

sns.kdeplot(song_data.popularity,song_data.loudness_db,shade=True,ax=ax[1][0]).set_title("Popularity to Loudness");

sns.kdeplot(song_data.popularity,song_data.liveness,shade=True,ax=ax[1][1]).set_title("Popularity to Liveness");

sns.kdeplot(song_data.popularity,song_data.valence,shade=True,ax=ax[1][2]).set_title("Popularity to Valence");

sns.kdeplot(song_data.popularity,song_data.length,shade=True,ax=ax[2][0]).set_title("Popularity to Length");

sns.kdeplot(song_data.popularity,song_data.acousticness,shade=True,ax=ax[2][1]).set_title("Popularity to Accousticness");

sns.kdeplot(song_data.popularity,song_data.speechiness,shade=True,ax=ax[2][2]).set_title("Popularity to Speechiness");