import numpy as np

import pandas as pd
df = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data.csv")

print(df.shape)

df.columns
df.dtypes
df.isna().sum().sum()
df.drop(['Unnamed: 0', 'id','explicit','key','release_date','mode'], axis=1, inplace=True)
df.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')



%matplotlib inline
corr = df[['acousticness','danceability','energy','instrumentalness','liveness','tempo','valence']].corr()



plt.figure(figsize=(12,8))

sns.heatmap(corr, annot=True)
df[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy', ascending=False)[:10]
df.acousticness.mean()
year_avg = df[['danceability','energy','liveness',

               'acousticness','valence','year']].groupby('year').mean().sort_values(by='year').reset_index()

year_avg.head()
plt.figure(figsize=(14,8))



plt.title("Song Trends Over Time", fontsize=15)

lines = ['danceability','energy','liveness','acousticness','valence']



for line in lines:

    ax = sns.lineplot(x='year', y=line, data=year_avg)



plt.legend(lines)
melted = year_avg.melt(id_vars='year')

melted.head()
print(len(melted))

print(len(year_avg))
plt.figure(figsize=(14,6))

plt.title("Song Trends Over Time", fontsize=15)

sns.lineplot(x='year', y='value', hue='variable', data=melted)
df.artists.nunique()
df.artists.value_counts()[:7]
artist_list = df.artists.value_counts().index[:7]

df_artists = df[df.artists.isin(artist_list)][['artists','year',

                                                          'energy']].groupby(['artists','year']).count().reset_index()

df_artists.rename(columns={'energy':'song_count'}, inplace=True)
df_artists.head()
plt.figure(figsize=(16,8))



sns.lineplot(x='year', y='song_count', hue='artists', data=df_artists)
df1 = pd.DataFrame(np.zeros((100,7)), columns=artist_list)

df1['year'] = np.arange(1921,2021)

print(df1.shape)

df1.head()
df1 = df1.melt(id_vars='year',var_name='artists', value_name='song_count')

print(df1.shape)

df1.head()
df_merge = pd.merge(df1, df_artists, on=['year','artists'], how='outer').sort_values(by='year').reset_index(drop=True)

df_merge.head()
df_merge.fillna(0, inplace=True)

df_merge.head()
df_merge.drop('song_count_x', axis=1, inplace=True)

df_merge.rename(columns={'song_count_y':'song_count'}, inplace=True)

df_merge.head()
df_merge['cumsum'] = df_merge[['song_count','artists']].groupby('artists').cumsum()



df_merge.head()
import plotly.express as px
fig = px.bar(df_merge,

             x='artists', y='cumsum',

            color='artists',

            animation_frame='year', animation_group='year',

            range_y=[0,1000],

            title='Artists with Most Number of Songs')

fig.show()