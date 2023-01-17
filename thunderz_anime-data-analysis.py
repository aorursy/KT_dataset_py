import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting
# matplotlib settings

import matplotlib



params = {'figure.dpi': 100,

          'figure.figsize': [10, 7.25],

          'font.size': 15.0}



matplotlib.rcParams.update(params)
# load data to df

anime_df = pd.read_csv('/kaggle/input/anime-dataset/anime_data.csv')

user_df = pd.read_csv('/kaggle/input/anime-dataset/user_data.csv')
# sort by rank and then drop it

anime_df = anime_df.sort_values(by='rank')

anime_df.drop(['rank'], axis=1, inplace=True)

anime_df.reset_index(inplace=True, drop=True) # index is now rank
# there are only finished anime in this dataset (everything before summer 2020)

anime_df.drop(['status'], axis=1, inplace=True)
anime_df.head()
anime_df['score'].plot(kind='hist')

plt.xlabel('scored from 1 to 10')

plt.show()
names = list(anime_df.sort_values(by='scored_by').tail(10)['title'])



anime_df['scored_by'].head(10).sort_values().plot(kind='bar')

plt.xticks([i for i in range(len(names))], names, rotation=90)

plt.show()
anime_df.sort_values(by='popularity').head(10)['title']
source = list(anime_df.groupby('source').score.mean().sort_values().index)



anime_df.groupby('source').score.agg(['mean']).sort_values(by='mean').plot(lw=2)

plt.grid()

plt.xticks([i for i in range(len(source))], source, rotation=90)

plt.xlabel('source')

plt.ylabel('mean score')

plt.legend()

plt.show()
anime_df.groupby('source').size().sort_values().plot(kind='bar')

plt.ylabel('count')

plt.xlabel('source')

plt.show()
anime_df.groupby('rating').size().sort_values().plot(kind='bar')

plt.show()
anime_df['aired_from'] = pd.to_datetime(anime_df.aired_from)

anime_df['aired_to'] = pd.to_datetime(anime_df.aired_to)



year_dist = anime_df[anime_df.aired_from.dt.year > 1995].aired_from.dt.year.value_counts().sort_index()

year_dist.iloc[:-1].plot(grid=True)

plt.show()
anime_df[(anime_df['episodes'] < 100) & (anime_df['type'] == 'TV')]['episodes'].plot(kind='hist', bins=40)

plt.xlabel('Number of episodes')

plt.show()
# outliers

anime_df[(anime_df['episodes'] > 100) & (anime_df['type'] == 'TV')].sort_values(by='episodes')['title'].head()
anime_df[(anime_df['episodes'] < 100) & (anime_df['type'] == 'OVA')]['episodes'].plot(kind='hist', bins=40)

plt.xlabel('Number of episodes')

plt.show()
import pickle 



# dict of all genres and their desriptions as values

file = open("/kaggle/input/anime-dataset/MAL_genres.pickle",'rb')

genres = pickle.load(file)

file.close()



genre_count = {genre: 0 for genre in genres.keys()}



for genre in genre_count.keys():

    genre_count[genre] = anime_df['genres'].apply(lambda x: x.count(genre)).sum()
genres_sorted = np.argsort(list(genre_count.values()))

x = np.array(list(genre_count.keys()))[genres_sorted]

y = np.array(list(genre_count.values()))[genres_sorted] / anime_df.shape[0]



pos = [i for i in range(len(genres))]



plt.bar(pos, y)

plt.xticks(pos, x, rotation=90, fontsize=12)

plt.ylabel('count / all')

plt.show()