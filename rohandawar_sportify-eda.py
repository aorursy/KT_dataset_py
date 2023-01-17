# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libs

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#read the data

songs = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

songs.head()
# seems that first Cols has indexing issues, so dropping the first Cols

songs.columns

songs.drop('Unnamed: 0', inplace=True, axis = 1)

songs.head()
#The cols has some whild characters such as '.' period & double period '..' so changing the same

songs.rename(columns={'Track.Name': 'track_name', 'Artist.Name':'artist_name', 'Genre':'genre',

                     'Beats.Per.Minute': 'beats_per_minute', 'Energy':'energy', 'Danceability':'danceability',

                     'Loudness..dB..':'loudness_db','Liveness':'liveness','Valence.':'valence', 'Length.':'length',

                     'Acousticness..':'acousticness', 'Speechiness.':'speechiness', 'Popularity':'popularity'}, inplace=True)

songs.head()
#checking the NULL values

songs.isnull().sum()
#checking the shape of the dataset

songs.shape
#checking the info for dataset

songs.info()
#Popularity Distribution analysis



plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

plt.title('Popularity Spread')

sns.boxplot(y='popularity', data=songs)



plt.subplot(1,2,2)

plt.title('Popularity Distribution')

sns.distplot(songs.popularity)

plt.show()
#Checking the Genre

plt.figure(figsize=(10,8))

sns.countplot(y='genre', data=songs)

plt.title('Genre count ')

plt.show()
plt.figure(figsize=(10,8))

plt1 = songs.artist_name.value_counts().plot('bar')

plt.title('Artist Name')

plt1.set(xlabel = 'Artist Name', ylabel='Frequency of Artist')

plt.show()
#summary Statics

songs.describe()
sns.pairplot(songs)

plt.show()
songs[songs.popularity>=95]
songs_genere = songs.groupby('genre')
songs_genere['artist_name'].unique()