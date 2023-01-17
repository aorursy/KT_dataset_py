# Import the libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

%matplotlib inline
# Import the data
df = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data.csv")

# View the shape and columns names
print(df.shape)
df.columns
# Check for missing values
df.isnull().sum()
# Drop unneccessary columns
df.drop(["id", "key", "mode", "explicit", "release_date"], axis=1, inplace=True)
df.head()
corr = df[["acousticness","danceability","energy", "instrumentalness", 
           "liveness","tempo", "valence", "loudness", "speechiness"]].corr()

plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True)
year_avg = df[["acousticness","danceability","energy", "instrumentalness", 
               "liveness","tempo", "valence", "loudness", "speechiness", "year"]].\
groupby("year").mean().sort_values(by="year").reset_index()

year_avg.head()
# Create a line plot
plt.figure(figsize=(14,8))
plt.title("Song Trends Over Time", fontdict={"fontsize": 15})

lines = ["acousticness","danceability","energy", 
         "instrumentalness", "liveness", "valence", "speechiness"]

for line in lines:
    ax = sns.lineplot(x='year', y=line, data=year_avg)
    
    
plt.ylabel("value")
plt.legend(lines)
# Check for the number of unique artists
df["artists"].nunique()
# Top 10 artists with most songs
df["artists"].value_counts()[:10]
artist_list = df.artists.value_counts().index[:10]

df_artists = df[df.artists.isin(artist_list)][["artists","year"]].\
groupby(["artists","year"]).size().reset_index(name="song_count")

df_artists.head()
plt.figure(figsize=(14,8))
sns.lineplot(x="year", y="song_count", hue="artists", data=df_artists)
top_artists = pd.DataFrame(np.zeros((100,10)), columns=artist_list)
top_artists['year'] = np.arange(1921,2021)
print(top_artists.shape)
top_artists.head()
top_artists = top_artists.melt(id_vars='year',var_name='artists', value_name='song_count')
print(top_artists.shape)
top_artists.head()
df_merged = pd.merge(top_artists, df_artists, on=['year','artists'], how='outer').\
sort_values(by='year').reset_index(drop=True)
df_merged.head()
df_merged.fillna(0, inplace=True)
df_merged.drop('song_count_x', axis=1, inplace=True)
df_merged.rename(columns={'song_count_y':'song_count'}, inplace=True)
df_merged.head()
df_merged['cumsum'] = df_merged[['song_count','artists']].groupby('artists').cumsum()
df_merged.head(10)
fig = px.bar(df_merged,
             x='artists', y='cumsum',
             color='artists',
             animation_frame='year', animation_group='year',
             range_y=[0,1300],
             title='Artists with Most Songs')
fig.show()