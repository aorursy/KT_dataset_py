# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import time
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
# data loading
ratings = pd.read_csv('../input/movielens-20m-dataset/rating.csv', parse_dates=[3])
movies = pd.read_csv('../input/movielens-20m-dataset/movie.csv')
tags = pd.read_csv('../input/movielens-20m-dataset/tag.csv', parse_dates=[3])
relevances = pd.read_csv('../input/movielens-20m-dataset/genome_scores.csv')
tagIDs = pd.read_csv('../input/movielens-20m-dataset/genome_tags.csv')

ratings100k = pd.read_csv('../input/movielens-latest-small/ratings.csv')
tags100k = pd.read_csv('../input/movielens-latest-small/tags.csv')
ratings100k.timestamp = pd.to_datetime(ratings100k.timestamp, unit='s')
tags100k.timestamp = pd.to_datetime(tags100k.timestamp, unit='s')

movies100k = pd.read_csv('../input/movielens-latest-small/movies.csv')
movies = pd.merge(movies, movies100k, how='outer')
movies = movies[~movies.duplicated(subset='movieId')]
import os
print(os.listdir('../input/movielens-20m-dataset'))
print('Dataset contains {:,} ratings from {:,} distinct users applied to {:,} movies.'\
      .format(len(ratings), ratings['userId'].nunique(), ratings['movieId'].nunique()))
print('Dataset contains data since {} until {}.'\
      .format(ratings['timestamp'].min().date(), ratings['timestamp'].max().date()))
user_ratings = ratings.groupby(by='userId')
d = user_ratings['rating'].count()
limit = 500
plt.hist(d[d<=limit], bins='fd')
plt.xlabel('number of rated movies')
plt.ylabel('number of users')
print(f'Only users with less than {limit} ratings are displayed ({len(user_ratings) - len(d[d<=limit]):,} users omitted).')
plt.show()
users_average = ratings.groupby('userId')['rating'].mean()
items_average = ratings.groupby('movieId')['rating'].mean()
plt.hist([users_average, items_average], histtype='step', density=True)
plt.xlabel('average rating for a movie / by a user')
plt.ylabel('number of movies / users')
plt.legend(['average rating given by a user', 'average rating of a movie'], loc=2)
plt.show()
genres = Counter()
for g in movies['genres']:
    genres.update(g.split('|'))
print('List of 10 most common genres: ', genres.most_common(10))
movie_ratings = ratings.groupby(by='movieId')
most_rated = movie_ratings['rating'].count().sort_values(ascending=False).head(10)
top_rated = movie_ratings['rating'].mean().where(movie_ratings['rating'].count() > 20).sort_values(ascending=False).head(10)
print(pd.merge(pd.DataFrame(most_rated), movies, on='movieId')[['title','rating']].rename(index=lambda x: x+1, columns={'rating': 'n. of ratings'}),'\n')
print(pd.merge(pd.DataFrame(top_rated), movies, on='movieId')[['title','rating']].rename(index=lambda x: x+1, columns={'rating': 'average rating'}))
print(f"Tags were given by {tags['userId'].nunique():,} users to {tags['movieId'].nunique():,} movies.")
print(f"Together, {tags['tag'].nunique():,} unique tags were given in a period since {tags['timestamp'].min().date()} until {tags['timestamp'].max().date()}.")
tag_tags = tags.groupby(by='tag')
tag_tags['movieId'].count().sort_values(ascending=False).head(10).rename('n. of movies')
relevances.sort_values(by='relevance').head(20).merge(movies, on='movieId', how='left').merge(tagIDs, on='tagId', how='left').rename(index=lambda x: x+1)[['tag', 'title', 'relevance']]
class Sampler():
    def sample_full(self):
        return ratings
    def sample_latest(self):
        return ratings100k
    ratings_rated = None
    def sample_rated(self):
        if self.ratings_rated is None:
            most_rated = ratings.groupby(by='movieId').count().sort_values(by='userId').tail(500).index
            self.ratings_rated = ratings[ratings.movieId.isin(most_rated)]
        return self.ratings_rated
    ratings_active = None
    def sample_active(self):
        if self.ratings_active is None:
            most_active = ratings.groupby(by='userId').count().sort_values(by='movieId').tail(1000).index
            self.ratings_active = ratings[ratings.userId.isin(most_active)]
        return self.ratings_active
    def sample_random(self, sample_size=100000):
        return ratings.sample(sample_size)
    samplers = {'full': sample_full, 'official_small': sample_latest, 'most_rated': sample_rated, \
                'most_active': sample_active, 'random': sample_random}

sampler = Sampler()
print(f"'rated' dataset contains {len(sampler.samplers['most_rated'](sampler)):,} ratings.")
print(f"'active' dataset contains {len(sampler.samplers['most_active'](sampler)):,} ratings.")
class Divider():
    @staticmethod
    def divide_time(data):
        div_time = data['timestamp'].quantile(0.8)
        train = data[data['timestamp'] <= div_time]
        test = data[data['timestamp'] > div_time].copy()
        return train, test
    @staticmethod
    def divide_users(data):
        u = np.random.choice(data.userId.unique(), int(data.userId.nunique()*0.2))
        train = data[~data.userId.isin(u)]
        test = data[data.userId.isin(u)].copy()
        return train, test
    @staticmethod
    def divide_ratings(data):
        rank = data.groupby('userId').timestamp.rank(method='first', ascending=False)
        train = data[rank > 10]
        test = data[rank <= 10].copy()
        return train, test
    dividers = {'time': divide_time.__func__, 'users': divide_users.__func__, 'last_ratings': divide_ratings.__func__}
print('Division by time.')
for s in sampler.samplers:
    print(f'Sampling type: {s}')
    data = sampler.samplers[s](sampler)
    train, test = Divider.dividers['time'](data)
    new_users = set(test['userId']) - set(train['userId'])
    new_items = set(test['movieId']) - set(train['movieId'])
    print('Test subset is {:,} ratings long. It contains {} days of data.\n\
Test subset contains {:,} new users who created {:.4} % of all test ratings.\n\
Test subset contains {:,} new movies that correspond to {:.4} % of all test ratings.'\
          .format(len(test), (test['timestamp'].max() - test['timestamp'].min()).days, \
                  len(new_users), len(test[test['userId'].isin(new_users)])/len(test)*100,\
                  len(new_items), len(test[test['movieId'].isin(new_items)])/len(test)*100))
test_set_sizes = pd.DataFrame(index=sampler.samplers, columns=Divider.dividers)
for s in sampler.samplers:
    for d in Divider.dividers:
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        test_set_sizes.at[s,d] = len(test)/len(data)*100
test_set_sizes
r = (4.5 * np.random.random_sample((len(test),))) + 0.5
def RMSE(true, predicted):
    return np.sqrt(MSE(true, predicted))
def RMSE1(true, predicted):
    return np.sqrt(((true - predicted) ** 2).mean())
start = time.time()
rmse = RMSE1(test['rating'], r)
print(f'took {time.time()-start} seconds for numpy. RMSE = {rmse}')
start = time.time()
rmse = RMSE(test['rating'], r)
print(f'took {time.time()-start} seconds for sklearn. RMSE = {rmse}')
methods = pd.MultiIndex.from_product([sampler.samplers, Divider.dividers], names=['sampling methods', 'division methods'])
models = ['random', 'user_avg', 'item_avg', 'item_CF', 'item_CF_lenskit']
results = pd.DataFrame(index=models, columns=methods)
results
start = time.time()
for s in sampler.samplers:
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        r = (4.5 * np.random.random_sample((len(test),))) + 0.5
        rmse = RMSE(test['rating'], r)
        results.loc['random',(s,d)] = rmse
print(f"Computation took {time.time() - start:.6} seconds.")
results
start = time.time()
for s in sampler.samplers:
    if s != 'official_small':  # for performance reasons when submitting
        continue
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        global_average = train['rating'].mean()
        users_average = train.groupby(by='userId')['rating'].mean()
        test['predicted'] = np.repeat(global_average, len(test))  # global average fallback
        users = np.intersect1d(users_average.index, test.userId.unique(), assume_unique=True)  # will be empty for users divider
        c = 0
        p = 0
        step = len(users)/10
        print('[__________]')
        for u in users:
            c += 1
            if c >= (p+1)*step:
                p += 1
                print('[' + '#'*p + '_'*(10-p) + ']')
            test.loc[test['userId'] == u,'predicted'] = users_average[u]
        rmse = RMSE(test['rating'], test['predicted'])
        results.loc['user_avg',(s,d)] = rmse
print(f"Computation took {time.time() - start:.6} seconds.")
results
start = time.time()
for s in sampler.samplers:
    if s != 'official_small':
        continue
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        global_average = train['rating'].mean()
        genres_average = train.merge(movies[['movieId', 'genres']]).groupby('genres').rating.mean()
        items_average = train.groupby(by='movieId')['rating'].mean()
        test['predicted'] = np.repeat(global_average, len(test))  # second fallback for unknown genres
        test_items = test.movieId.unique()
        c = 0
        p = 0
        step = len(test_items)/10
        print('[__________]')
        for i in np.intersect1d(test_items, items_average.index, assume_unique=True):  # predict item average
            c+=1
            if c >= (p+1)*step:
                p += 1
                print('[' + '#'*p + '_'*(10-p) + ']')
            test.loc[test['movieId'] == i,'predicted'] = items_average[i]
        for i in np.setdiff1d(test_items, items_average.index, assume_unique=True):  # predict genre average for new items
            c+=1
            if c >= (p+1)*step:
                p += 1
                print('[' + '#'*p + '_'*(10-p) + ']')
            g = movies.loc[movies.movieId == i, 'genres']
            if g.empty:  # unknown genre
                continue
            try:
                a = genres_average[g.values[0]]
            except KeyError:  # not a perfect genres match
                a = genres_average.filter(like=g.values[0])  # try any more specific genres
                if a.empty:
                    a = global_average
                    for j in g.values[0].split('|'):  # try subgenres
                        try:
                            a = (a + genres_average[j]) / 2
                        except KeyError:
                            continue
                else:
                    a = a.mean()
            test.loc[test['movieId'] == i, 'predicted'] = a
        rmse = RMSE(test['rating'], test['predicted'])
        results.loc['item_avg',(s,d)] = rmse
print(f"Computation took {time.time() - start:.6} seconds.")
results
for s in sampler.samplers:
    if s != 'official_small':
        continue
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        new_items = set(test['movieId']) - set(train['movieId'])
        print("Number of new items:{}\nData from test set created by new items: {:.4} %".format(len(new_items), len(test[test['movieId'].isin(new_items)])/len(test)*100))
def itemCFpred(user, item, user_c, item_c, item_user, matrix_full):
    i = np.argwhere(item_c.categories == item)[0][0]
    u = np.argwhere(user_c.categories == user)[0][0]
    rated = matrix_full.loc[:,matrix_full.loc[user,:].notna()].columns
    irated = np.argwhere(item_c.categories.isin(rated)).flatten()
    a = cos_sim(item_user[i,:],item_user[irated,:])[0]
    k = min(len(rated), K)
    ind = np.argpartition(a, -k)[-k:]
    similarities = a[ind]
    s = similarities.sum()
    r = np.multiply(item_user[irated, u][ind].todense(), similarities.reshape(k,1)).sum()
    return r/s
K = 50

start = time.time()
for s in sampler.samplers:
    if s != 'official_small':
        continue
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        global_average = train['rating'].mean()
        users_average = train.groupby(by='userId')['rating'].mean()
        items_average = train.groupby(by='movieId')['rating'].mean()
        genres_average = train.merge(movies[['movieId', 'genres']]).groupby('genres').rating.mean()
        # matrix_full = train.pivot(index='userId', columns='movieId', values='rating')  # too large
        user_c = CategoricalDtype(sorted(train['userId'].unique()), ordered=True)
        item_c = CategoricalDtype(sorted(train['movieId'].unique()), ordered=True)

        row = train['userId'].astype(user_c).cat.codes
        col = train['movieId'].astype(item_c).cat.codes
        user_item = csr_matrix((train["rating"], (row, col)), \
                                   shape=(user_c.categories.size, item_c.categories.size))
        item_user = csr_matrix((train["rating"], (col, row)), \
                                   shape=(item_c.categories.size, user_c.categories.size))
        matrix_full = pd.SparseDataFrame(user_item, index=user_c.categories, columns=item_c.categories)
        c = 0
        p = 0
        step = len(test)/50
        print('[__________________________________________________]')
        for row in test.itertuples():
            c += 1
            if c >= (p+1)*step:
                p += 1
                print('[' + '#'*p + '_'*(50-p) + ']')
            if row.movieId not in matrix_full.columns:  # unseen item
                try:
                    ua = users_average.at[row.userId]
                    test.at[row.Index, 'predicted'] = ua
                except KeyError:  # and unseen user
                    g = movies.loc[movies.movieId == i, 'genres']
                    if g.empty:  # unknown genre
                        test.at[row.Index, 'predicted'] = global_average                        
                    try:
                        a = genres_average[g.values[0]]
                    except KeyError:  # not a perfect genres match
                        a = genres_average.filter(like=g.values[0])  # try any more specific genres
                        if a.empty:
                            a = global_average
                            for j in g.values[0].split('|'):  # try subgenres
                                try:
                                    a = (a + genres_average[j]) / 2
                                except KeyError:
                                    continue
                        else:
                            a = a.mean()
                    test.at[row.Index, 'predicted'] = a
            elif row.userId not in matrix_full.index:  #seen item but unseen user
                test.at[row.Index, 'predicted'] = items_average[row.movieId]
            else:  # seen both user and item; use CF
                test.loc[row.Index, 'predicted'] = itemCFpred(row.userId, row.movieId, user_c, item_c, item_user, matrix_full)
        test.predicted.fillna(global_average, inplace=True)  # to be fail-proof
        rmse = RMSE(test['rating'], test['predicted'])
        results.loc['item_CF',(s,d)] = rmse
print(f"Computation took {time.time() - start:.6} seconds.")
results
from lenskit import batch
from lenskit.algorithms import item_knn as knn

algo = knn.ItemItem(K)

start = time.time()
for s in sampler.samplers:
    if s != 'official_small':
        continue
    for d in Divider.dividers:
        print(f'Computing {s} sampler and {d} divider.')
        data = sampler.samplers[s](sampler)
        train, test = Divider.dividers[d](data)
        train.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        test.rename(columns={'userId': 'user', 'movieId': 'item'}, inplace=True)
        global_average = train['rating'].mean()
        print('training')
        model = algo.train(train)
        print('recommending')
        recs = batch.predict(algo, test[['user', 'item']], model)
        res = pd.merge(recs, test, how='left', on=('user', 'item'))
        res.prediction.fillna(global_average, inplace=True)
        rmse = RMSE(test['rating'], res['prediction'])
        results.loc['item_CF_lenskit',(s,d)] = rmse
print(f"Computation took {time.time() - start:.6} seconds.")
results
