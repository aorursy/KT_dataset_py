import numpy as np
import pandas as pd 
import os
from tqdm import tqdm
import dask.bag as dask_bag
import glob
import dask.dataframe as dask_df

print(os.listdir("../input"))
for idx, df in tqdm(enumerate(pd.read_csv('../input/rating.csv', chunksize = 500000))):
    df.to_csv('rating_k{}.csv.gz'.format(idx), compression = 'gzip', index = False)
#reads whole file
pd.read_csv('../input/movie.csv')
#reads 10 rows from file
pd.read_csv('../input/movie.csv', nrows = 10)
#reads whole file skipping 10 first rows
pd.read_csv('../input/movie.csv', skiprows = 10)
#iterate over file with 10 rows chunks
pd.read_csv('../input/movie.csv', chunksize = 10)
#as we can see we dont have to do anything to handle compressed files
files_to_read = glob.glob('rating_k*.csv.gz')
print('Files to load: {}'.format(len(files_to_read)))
%%time
dfs = []
for f in files_to_read:
    dfs.append(pd.read_csv(f))
%%time
bag = dask_bag.from_sequence(files_to_read)
dfs = bag.map(lambda f: pd.read_csv(f)).compute(scheduler='threads')
%%time
ratings = pd.DataFrame()
for df in dfs:
    ratings = ratings.append(df, ignore_index = True)
print("Ratings shape: {}".format(ratings.shape))
%%time
ratings = pd.concat(dfs, ignore_index = True)
print("Ratings shape: {}".format(ratings.shape))
ratings.head(10)
print('Ratings rows number: {}'.format(len(ratings)))
print('Ratings have NaN values: {}'.format(ratings.isna().values.any()))
print('Ratings have null values: {}'.format(ratings.isnull().values.any()))
ratings.dtypes
ratings.nunique()
ratings['rating'].describe()
%%time
ratings['timestamp_dt'] = ratings['timestamp'].map(pd.to_datetime)
%%time
ddf = dask_df.from_pandas(ratings, npartitions=4)
ratings['timestamp_dt'] = ddf['timestamp'].map(np.datetime64, meta = pd.Series(dtype=np.dtype(str))).compute()

%%time
ratings['year'] = ratings.apply(lambda x: x['timestamp_dt'].year, axis = 1)
ratings['month'] = ratings.apply(lambda x: x['timestamp_dt'].month, axis = 1)
ratings['day_of_week'] = ratings.apply(lambda x: x['timestamp_dt'].day_name(), axis = 1)
%%time
ratings['year'] = ratings['timestamp_dt'].map(lambda x: x.year)
ratings['month'] = ratings['timestamp_dt'].map(lambda x: x.month)
ratings['day_of_week'] = ratings['timestamp_dt'].map(lambda x: x.day_name())
# del ratings['timestamp']
ratings = ratings.drop(columns = ['timestamp'])
ratings.head()
%%time
by_movieId = ratings.groupby(['movieId'])['rating'].agg(['count', 'mean', 'max', 'min'])
by_movieId.head()
top_20_ranked = by_movieId.sort_values(by = ['count'], ascending  = False).iloc[0:20]
%%time
movies = pd.read_csv('../input/movie.csv')
top_20_ranked = pd.merge(left = top_20_ranked, right = movies, on = ['movieId'], how = 'inner')
top_20_ranked
top_20_ranked.plot.bar(x = 'title', y = 'count', figsize = (20, 7))
#group by movieId and calculate mean and count
df = ratings.groupby(['movieId'])['rating'].agg(['mean', 'count'])
#drop movies without votes number less than 500
df = df.loc[df['count'] > 1000]
#sort to get top 5
df = df.sort_values(by = 'mean', ascending = False)
df = df.iloc[0:5]
#merge with titles
df = pd.merge(left = df, right = movies, on=['movieId'], how = 'inner')
df
#filter movies with correct titles
filtered_movies = movies.loc[(movies['title'].str.contains('Harry Potter') | movies['title'].str.contains('Star Wars'))]
#merge with titles
df = pd.merge(left = ratings, right = filtered_movies, on=['movieId'], how = 'inner')
#group by title and calculate mean and count
df = df.groupby(['title'])['rating'].agg(['mean', 'count'])
# add a feature for sorting purposes
df['production_year'] = df.index.map(lambda x: int(x[-5:-1]))
df['m'] = df.index.map(lambda x: x[0])
#drop movies without votes number less than 500 because we collect some documentaries about StarWars
df = df.loc[df['count'] > 500]
#sort to show trend on plot
df = df.sort_values(by = ['m', 'production_year'])
#plot
df.plot.bar(y = 'mean', figsize = (20, 7))
#filter movies with correct titles
filtered_movies = movies.loc[movies['title'].str.contains('Star Wars')]
#merge with titles
df = pd.merge(left = ratings, right = filtered_movies, on=['movieId'], how = 'inner')
#group by title and calculate mean and count
df = df.groupby(['title', 'year'])['rating'].agg(['mean', 'count'])
#drop movies without votes number less than 500 because we collect some documentaries about StarWars
df = df.loc[df['count'] > 500]
#sort to show trend on plot
df = df.sort_values(by = 'year')
#plot
df.unstack(level = 0).plot(y = 'mean', figsize = (20, 7))
