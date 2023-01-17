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
import seaborn as sns                       

import matplotlib.pyplot as plt             

%matplotlib inline     

sns.set(color_codes=True)
df = pd.read_csv("/kaggle/input/spotify-global-2019-moststreamed-tracks/spotify_global_2019_most_streamed_tracks_audio_features.csv")

df.head(5)
df.tail(5)
df.dtypes
new_df = df.drop(['Country', 'Track_id', 'URL', 'Artist_id', 'Artist_img'], axis=1)

new_df.head(10)
f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(new_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
print("Mean value for danceability:", new_df['danceability'].mean())

sns.distplot(new_df['danceability'])

plt.show()

print("Mean value for energy:", new_df['energy'].mean())

sns.distplot(new_df['energy'])

plt.show()

print("Mean value for mode:", new_df['mode'].mean())

sns.distplot(new_df['mode'])

plt.show()

print("Mean value for speechiness:", new_df['speechiness'].mean())

sns.distplot(new_df['speechiness'])

plt.show()

print("Mean value for acousticness:", new_df['acousticness'].mean())

sns.distplot(new_df['acousticness'])

plt.show()

print("Mean value for instrumentalness:", new_df['instrumentalness'].mean())

sns.distplot(new_df['instrumentalness'])

plt.show()

print("Mean value for liveness:", new_df['liveness'].mean())

sns.distplot(new_df['liveness'])

plt.show()

print("Mean value for valence:", new_df['valence'].mean())

sns.distplot(new_df['valence'])

plt.show()
numeric = new_df.drop(['Rank','Streams','Artist_popularity', 'Artist_follower'], axis=1)

small = numeric.drop(['tempo','duration_ms','key', 'loudness', 'time_signature'], axis=1)

sns.set_palette('pastel')

small.mean().plot.bar()

plt.title('Mean Values of Audio Features')

plt.show()