%cd D:\Python Projects\Imdb
import pandas as pd

import numpy as np
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
rating = pd.read_csv('../input/rating.tsv', sep = '\t')

epi = pd.read_csv('../input/episode.tsv', sep = '\t')

all1 = pd.read_csv('../input/everything.tsv', sep = '\t')
rating.head()
rating.info()
epi.head()
all1.head()
all1.info()
rating.describe()
print(rating.numVotes.min())
diff = rating.numVotes.max() - rating.numVotes.min()

print(diff)
rating.columns = ["tconst", "avg_rating", "votes"]

rating.columns
c_prev = len(rating.votes)

print(c_prev)

rating = rating.loc[rating['votes'] > 1000]

print(len(rating))
epi.describe()
epi.columns = ['tconst', 'p_tconst', 'season', 'episode']
epi = epi.replace('\\N', '0.00001')

epi.head()
epi['season'] = epi['season'].astype(float)
epi['episode'] = epi['episode'].astype(float)
epi.info()
print(len(epi))
epi = epi[(epi['season'] != 0.00001) | (epi['episode'] != 0.00001)]
print(len(epi))
epi.head()
all1 = all1.drop('originalTitle', axis = 1)
all1.columns = ['tconst', 'type', 'title', 'adult', 'start', 'end', 'runtime', 'genres']
all1.info()
all1.head()
all1.tail()
all1.replace('\\N', '0.000000001')
all1.head()
#all1['runtime'] = all1['runtime'].astype(float)

all1 = all1.drop('end', axis = 1)

all1.head()
all_clean = all1[all1['runtime'] != '\\N']

all_clean = all_clean[all_clean['genres'] != '\\N']

all_clean.head()
print(len(all1))

print(len(all_clean))
all_clean = all_clean.dropna(subset = ['genres'])

all_clean.head()

split_genres = pd.DataFrame(all_clean['genres'].str.split(',').tolist(), columns = ['gen1', 'gen2', 'gen3'])

split_genres.head()
split_genres.shape
all_clean.shape
all_clean = all_clean.join(split_genres)

all_clean.head()
all_clean.head(10) #delete
x = all_clean[all_clean['type'] == 'tvEpisode'] #delete

x
#all_clean = all_clean.drop('genres', axis = 1)

all_clean.head()
genres_1 = split_genres.gen1.value_counts()

genres_1.head()
genres_2 = split_genres.gen2.value_counts()

genres_3 = split_genres.gen3.value_counts()

genres_2.head()
genres_3.head()
genres_all = pd.concat([genres_1, genres_2, genres_3], axis = 1, sort = False)
genres_all.head()
genres_all['index1'] = genres_all.index
genres_all = genres_all.fillna(0)
genres_all
genres_all['gen_sum'] = genres_all.gen1 + genres_all.gen2 + genres_all.gen3

genres_all.head()
rating.columns
epi.columns
all1.columns
epi_rating = rating.merge(epi, on = 'tconst', how = 'left')

print(len(epi_rating))
epi_rating.head()
epi_rating.info()
epi_rating.dtypes
epi_rating_notnull = epi_rating[epi_rating['p_tconst'].notnull()]
epi_rating_notnull.head()
final_rating = epi_rating_notnull.copy()
final_rating['rating'] = (final_rating['avg_rating']*final_rating['votes'])
final_rating.head()
final_rating.sort_values(by = 'rating', ascending = False)
parent =  final_rating['p_tconst'].unique()
top_episode = {}

for item in parent:

    selection = final_rating[final_rating['p_tconst'] == item]

    sorted1 = selection.sort_values('rating', ascending = False)

    top1 = sorted1.iloc[0]

    top_episode[item] = top1    
len(top_episode)
import pprint
pprint.pprint(top_episode)
col = ['tconst', 'p_tconst', 'avg_rating', 'votes', 'season', 'rating']

top_df = pd.DataFrame.from_dict(top_episode, orient = 'index', columns = col)
top_df.head() #top episodes from all the serials
top_df.tail()
top_df.sort_values(by = 'rating', ascending = False)
topdf = pd.merge(left = top_df, right = all1, on = 'tconst', how = 'left')

top_df = topdf.copy()

top_df.head()
def split_genres(series_col):

    split_col = pd.DataFrame(series_col.str.split(',').tolist(), columns = ['gen1', 'gen2', 'gen3'])

    return split_col
top_df_genres = split_genres(top_df.genres)

top_df_genres.head()
top_df = top_df.join(top_df_genres)
top_df = top_df.drop(['genres'], axis = 1)

top_df.head()
top_50 = top_df.sort_values(by = 'rating', ascending = False)[:50] #top 50 episodes from serials
top_50
top_50.info()
top_50['start'] = top_50['start'].astype(int)

top_50['runtime'] = top_50['runtime'].astype(int)
top_50.info()
top_50.describe()
top_50.start.value_counts().sort_values()
oldest_episode = top_50[top_50['start'] == 1989]

oldest_episode
latest_episode = top_50[top_50['start'] == 2019]

latest_episode
def gen_count(g1, g2, g3):

    

    gn1 = g1.value_counts()

    gn2 = g2.value_counts()

    gn3 = g3.value_counts()

    combined = pd.concat([gn1,gn2,gn3], axis = 1, sort = False)

    combined = combined.fillna(0)

    combined['index1'] = combined.index

    combined['gen_sum'] = combined.gen1 + combined.gen2 + combined.gen3

    gen_sum = combined[['index1', 'gen_sum']]

    gen_sum = gen_sum.sort_values(by = 'gen_sum', ascending = False)

    return gen_sum
gen_topdf = gen_count(top_df.gen1, top_df.gen2, top_df.gen3)

gen_topdf
gen_top50 = gen_count(top_50.gen1, top_50.gen2, top_50.gen3)

gen_top50
top_df.shape
ax = top_50[:5].plot(x = 'title', y = 'avg_rating', kind = 'bar', legend = False, ylim = (0,12), title = 'Rating of Top 5 Episodes (All-Time)')

ax.set_xlabel('')
fig, ax = plt.subplots(figsize = (20,5), nrows = 1, ncols = 2)

gen_topdf.plot(x = 'index1', y = 'gen_sum', kind = 'bar', legend = False, title = 'Serial-wise Genres: All', ax = ax[0])

gen_top50.plot(x = 'index1', y = 'gen_sum', kind = 'bar', legend = False, title = 'Serial-wise Genres: Top 50 Episodes', ax = ax[1])
all1.shape
all_clean.shape
all_clean.head(9)
all1.type.unique()
all1[all1['type'] == 'movie'].shape
movies = all1[all1['type'] == 'movie']
movies.head()
movies.info()
#movies = movies.drop('adult', axis = 1)

#movies.runtime = movies.runtime.astype(int)

#movies.info()
movies.start = movies.start.replace('\\N', 111111)

movies['start'] = movies.start.astype(int)

movies.start.min()
movies = movies[movies['start'] < 2019]
movies.start.max()
movies = movies.merge(rating, on = 'tconst', how = 'left')

movies.head()
movies_gen = split_genres(movies.genres)

movies_gen
movies = movies.join(movies_gen)

movies.head()
movies = movies.dropna(subset = ['avg_rating'])

movies.shape
movies.head()
movies['ratings'] = movies['avg_rating']*movies['votes']

movies.head()
movies_sort = movies.sort_values(by = 'ratings', ascending = False)

movies_sort.head()
movies_top50 = movies_sort[:50]

movies_top50.shape
movies_top50
movies_top50[:5]
ax = movies_top50[:5].plot(x = 'title', y = 'avg_rating', kind = 'bar', legend = False, ylim = (0,10), title = 'Rating of Top 5 Movies (All-Time)')

ax.set_xlabel('')
movies_sort_gen = gen_count(movies_sort.gen1, movies_sort.gen2, movies_sort.gen3)

movies_sort_gen
moviestop50_gen = gen_count(movies_top50.gen1, movies_top50.gen2, movies_top50.gen3)

moviestop50_gen
fig, ax = plt.subplots(figsize = (15,5), nrows = 1, ncols = 2)

movies_sort_gen.plot(x = 'index1', y = 'gen_sum', kind = 'bar', legend = False, title = 'Movie-wise Genres: All', ax =ax[0])

moviestop50_gen.plot(x = 'index1', y = 'gen_sum', kind = 'bar', legend = False, title = 'Movie-wise Genres: Top 50 Movies', ax = ax[1])