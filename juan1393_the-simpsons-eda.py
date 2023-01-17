import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_episodes = pd.read_csv('../input/simpsons_episodes.csv')
df_episodes.head()
df_episodes['us_viewers_in_millions'].isnull().values.any()
df_episodes = df_episodes[df_episodes['us_viewers_in_millions'].notnull()]

df_episodes = df_episodes[df_episodes['imdb_rating'].notnull()]
seasons = np.arange(1, len(df_episodes['season'].unique()))

visualizations_in_millions_per_season = df_episodes.groupby(['season'])['us_viewers_in_millions'].mean()

visualizations_in_millions_per_season.plot()

plt.xticks(seasons)

plt.show()
imdb_rating_per_season = df_episodes.groupby(['season'])['imdb_rating'].mean()

imdb_rating_per_season.plot()

plt.xticks(seasons)

plt.show()
from sklearn.preprocessing import MinMaxScaler



sc = MinMaxScaler()

df_episodes['us_viewers_in_millions_scaled'] = sc.fit_transform(df_episodes['us_viewers_in_millions'].values.reshape(-1,1))

df_episodes['imdb_rating_scaled'] = sc.fit_transform(df_episodes['imdb_rating'].values.reshape(-1,1))



visualizations_in_millions_per_season = df_episodes.groupby(['season'])['us_viewers_in_millions_scaled'].mean()

imdb_rating_per_season = df_episodes.groupby(['season'])['imdb_rating_scaled'].mean()

visualizations_in_millions_per_season.plot(label='US viewers in millions')

imdb_rating_per_season.plot(label='IMDB rating')

plt.xticks(seasons)

plt.legend()

plt.show()
highest_imdb_rating = df_episodes['imdb_rating'].max()

df_episodes.loc[df_episodes['imdb_rating'] == highest_imdb_rating]
matrix_imdb_rating = np.zeros((seasons.size, df_episodes['number_in_season'].max()))



for season in range(1, seasons.size):

    df_episodes_season = df_episodes[df_episodes['season'] == season]

    for episode in np.nditer(df_episodes_season['number_in_season'].values):

        matrix_imdb_rating[season - 1, episode - 1] = df_episodes_season.loc[df_episodes_season['number_in_season'] == episode]['imdb_rating']



ax = plt.subplot(1, 1, 1)

c = plt.pcolor(matrix_imdb_rating.T, cmap='viridis')

ax.set_xlabel('Seasons')

ax.set_ylabel('Episodes')

plt.show()