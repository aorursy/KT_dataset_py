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
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
df = pd.read_csv("../input/top50spotify2019/top50.csv", encoding='ISO-8859-1')

#To get an idea of the dataset that we are dealing with

df.head()
df.shape
df.info()
# Dropping 'Unnamed: 0' since it doesn't consist of any relevant information

df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.rename(columns={'Track.Name':'Track_Name', 

                   'Artist.Name':'Artist_Name',

                   'Beats.Per.Minute':'Beats_Per_Minute', 

                   'Loudness..dB..':'Loudness',

                   'Valence.':'Valence', 

                   'Length.':'Length', 

                   'Acousticness..':'Acousticness',

                   'Speechiness.':'Speechiness'}, inplace=True)
#To see our data after dropping Unnamed Column

df.head()
#We want to see how each song is being measured by

df.describe().T
df.Genre.value_counts()
plt.style.use('fivethirtyeight')

plt.figure(figsize = (16,10));

sns.countplot(x="Genre", data=df, linewidth=2, edgecolor='black');

plt.ylabel('Number of occurances');

plt.xticks(rotation=45, ha='right');
df.Artist_Name.value_counts()
plt.figure(figsize=(15,8))

plt.style.use('fivethirtyeight')

sns.countplot(x=df['Artist_Name'],data=df, linewidth=2, edgecolor='black')

plt.title('Number of times an artist appears in the top 50 songs list')

plt.xticks(rotation=50, ha='right')

plt.show()

#We want to interpret the data of the top artists

top_artists = df.groupby('Artist_Name')

filtered_data = top_artists.filter(lambda x: x['Artist_Name'].value_counts() > 1)
plt.figure(figsize=(20,8))

plt.style.use('fivethirtyeight')

sns.countplot(y=filtered_data['Artist_Name'],data=filtered_data, linewidth=2, edgecolor='black', order=filtered_data["Artist_Name"].value_counts().index)

plt.title('Top Artists of 2019')

plt.xticks(rotation=45, ha='right')

plt.show()


# The data set contains the following fields:



# Track.Name — Name of Track

# Artist.Name — Name of the Artist

# Genre — Genre of Track

# Beats.Per.Minute — Tempo of the Song

# Energy — The energy of Song — the higher the value the more energetic

# Danceability — Thee higher the value, the easier it is to dance to the song

# Loudness..dB.. — The higher the value, the louder the song.

# Liveness — The higher the value, the more likely the song is a live recording.

# Valence. — The higher the value, the more positive mood for the song.

# Length. — The duration of the song.

# Acousticness.. The higher the value the more acoustic the song

# Speechiness. — The higher the value the more spoken word the song contains

# Popularity — The higher the value the more popular the song is.



# We want to see if there is a strong correlation between how songs are 

correlations = df.corr()



fig = plt.figure(figsize=(12, 8))

sns.heatmap(correlations, annot=True, linewidths=1, cmap='YlGnBu', center=1)

plt.show()
