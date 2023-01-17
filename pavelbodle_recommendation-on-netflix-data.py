import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dask.dataframe as dd # asynchronous load and subset which is useful for large dataset sampling
from dask.distributed import Client

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid", context='paper')
sns.set(rc={'figure.figsize':(18,5)})

from functools import wraps # a ditty decorator

from sklearn.metrics.pairwise import cosine_similarity

import random # random integers
import os # i/o read files
import time # time my work
import gc  # clear ram
print(os.listdir("../input"))
print(os.listdir())
def time_this(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{} takes {} minutes!".format(func.__name__, round((end-start)/60, 2)))
        gc.collect()
        return result 
    return wrapper
wd = '../input/netflix-prize-data/' # working dir
data_files = [wd + 'combined_data_{}.txt'.format(i) for i in range(1,5)]
print('Data files to be combined and pre-processed: ')
print(data_files)
get_movie_id = lambda line: int(line.replace(':', '').replace('\n', ''))
get_rating = lambda line: [x.replace('\n', '') for x in line.split(',')]
df_it = lambda row: pd.DataFrame(row, index=['cust_id', 'rating', 'date']).T

def append_to_csv(data, fp):
    print('Writing {} rows to {}'.format(data.shape[0], fp))
    if os.path.exists(fp):
        data.to_csv(fp, mode='a', index=False, header=False)
    else:
        data.to_csv(fp, index=False)

@time_this
def get_ratings(fp):
    """
    Parse the text files that have movie id and customer ratings into a usable dataframe.
    @fp: file path (str)
    """
    print('Getting ratings from file: {}'.format(fp))
    agg_data = []
    with open(fp, 'r') as file_:
        for line_number, line in enumerate(file_):
            if (line_number % 10**7) == 0: print('{} million rows..'.format(line_number / (10**6)))
            if ':' in line:
                movie_id = get_movie_id(line)                      
            else:
                rating_row = get_rating(line)
                rating = {str(col): val for col,val in enumerate(rating_row)}
                rating['movie_id'] = movie_id
                agg_data.append(rating)            
    agg_data = pd.DataFrame(agg_data)
    agg_data.rename(columns={'0': 'cust_id', '1': 'rating', '2': 'date'}, inplace=True)
    print('Finished getting ratings from file: {}'.format(fp))
    return agg_data
#for fp in data_files:
df = get_ratings(data_files[0])
append_to_csv(df, 'ratings.csv')
del df
gc.collect()
os.listdir()
data = dd.read_csv('ratings.csv')
print(data.shape)
print(data.head())
n = random.randint(0, 30)
data = data[data.cust_id % n == 0]
client = Client()   # initialize the cluster
data = client.persist(data)
data = data.compute()
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
print(data.shape)
movie_titles = pd.read_csv(wd + 'movie_titles.csv',
                           encoding = 'ISO-8859-1', # some weird encoding issue
                           header = None, names = ['movie_id', 'year', 'name'])
movie_titles.drop_duplicates(subset=['name', 'year'], inplace=True)
movie_meta = pd.read_csv('../input/the-movies-dataset/movies_metadata.csv', 
                         low_memory=False,
                         usecols=['adult', 'budget', 'original_title',
                                 'release_date', 'popularity', 'revenue', 'genres'])
movie_meta.dropna(how='any', inplace=True)
movie_meta['release_date'] = pd.to_datetime(movie_meta['release_date'])
movie_meta['meta_year'] = movie_meta['release_date'].dt.year
movie_meta.drop_duplicates(subset=['original_title', 'meta_year'], inplace=True)
movie_titles.sample(5)
movie_meta = movie_titles.merge(movie_meta, how='left', left_on=['name', 'year'], right_on=['original_title', 'meta_year'],
                  validate='1:1', indicator=True).sort_values('_merge', ascending=False).query("_merge == 'both'")
sns.set(rc={'figure.figsize':(15,4)})
sns.set_style('whitegrid')
grouped = movie_titles.groupby('year', as_index=False)['movie_id'].count()
ax = sns.lineplot(x='year', y='movie_id', data=grouped, color='red')
ax.set_title('Movies by Release Date')
ax.set_xlabel('Year')
ax.set_ylabel('Movie Count')
data = data.merge(movie_titles, how='left', on='movie_id',validate='m:1', suffixes=['', '_of_release'])
data.sample(5)
agg = {
    'cust_id': 'nunique',
    'rating': {'min', 'max', 'mean', 'count'}
}
ratings = data.groupby(['name', 'year_of_release'], as_index=False).agg(agg)
ratings.columns = ['_'.join(col).strip('_') for col in ratings.columns.values]
ratings.rename(columns={'cust_id_cust_id': 'unique_customers'}, inplace=True)
ratings.sample(5)
ax = ratings.query('rating_count > 300').plot(kind='scatter', x='year_of_release', y='rating_mean', 
                                              color='red', s=4)
ax.set_ylabel('Average Rating')
ax.set_xlabel('Year of Movie Release')
plt.annotate('* Subset to Movies with >300 Ratings', (.75,0), (0,-40), xycoords='axes fraction', 
             textcoords='offset points', va='top', fontsize=10)
ax.set_title('Movie Ratings by Year of Release')
top = ratings.query('rating_count > 300').sort_values('rating_mean').tail(10)[['name', 'rating_mean']]
bottom = ratings.query('rating_count > 300').sort_values('rating_mean').head(10)[['name', 'rating_mean']]
fig, axes = plt.subplots(2, figsize=(15,7), sharex=True)
ax1 = plt.subplot(2, 1, 1)
top.set_index('name').plot(kind='barh', color='red', ax=ax1, legend=False)
ax1.set_ylabel('')
ax1.set_xlim(0,5)
ax1.xaxis.set_visible(False)
ax2 = plt.subplot(2, 1, 2)
bottom.set_index('name').plot(kind='barh', color='blue', ax=ax2, legend=False, position=1)
ax2.set_xlabel('User Rating')
ax2.set_ylabel('')
ax2.set_xlim(0,5)
plt.annotate('* Subset to Movies with >300 Ratings', (.75,0), (0,-40), xycoords='axes fraction', 
             textcoords='offset points', va='top', fontsize=10)
fig.suptitle("Highest and Lowest Rated Movies", fontsize=16)
fig, axes = plt.subplots(2, figsize=(20,7))
fig.suptitle("User Ratings & Volume Over Time", fontsize=16)
ax = plt.subplot(1, 2, 1)
grouped = data.groupby('year', as_index=False).agg(
    {'rating': 'mean', 'name': 'nunique', 'cust_id': 'count'})\
.rename(columns={'name': 'number of unique movies', 'cust_id': 'volume of reviews'})
grouped.plot(x='year', y=['rating','volume of reviews'], 
             secondary_y=['volume of reviews'], color=['red', 'blue'], ax=ax, grid=True)
ax.set_xlabel('')
ax2 = plt.subplot(1, 2, 2)
grouped.plot(x='year', y='number of unique movies', ax=ax2, color='red')
ax2.set_xlabel('')
min_movie = 1000   # movie has to have been rated over 1000 times
min_user = 200   # user has to have rated at least 200 times
users = data.groupby('cust_id')['rating'].count()
users = users.loc[users > min_user].index.values
movies = data.groupby('movie_id')['rating'].count()
movies = movies.loc[movies > min_movie].index.values
filtered = data.loc[data.cust_id.isin(users) & data.movie_id.isin(movies)]
print('Unfiltered: ', data.shape[0])
print('Filtered: ', filtered.shape[0])
print('Kept {}% of data'.format(round(filtered.shape[0]/data.shape[0], 2)*100))
filtered.sample(5)
mat = filtered.pivot_table(index='cust_id', columns='movie_id', values='rating')
print('The User-Movie Matrix')
mat.sample(10)
means = filtered.groupby('movie_id')['rating'].mean().to_dict()   # get a lookup table of movie to mean rating
topNrecs = mat.copy(deep=True)
for col in topNrecs:                    # for each movie
    already_rated = topNrecs[col].notnull()    # make note of which ones theyve already rated
    topNrecs[col].fillna(means[col], inplace=True)    # fill out the mean rating for each movie
    topNrecs.loc[already_rated, col] = np.nan       # remove the information we already have
print('Average User Rating Imputed onto Users Matrix')
topNrecs.sample(10)
recommendations = topNrecs.stack()\
.reset_index()\
.rename(columns={0: 'imputed_rating'})\
.groupby('cust_id')\
.apply(lambda x: x.nlargest(5, columns='imputed_rating'))\
.reset_index(drop=True)\
.sort_values(by=['cust_id', 'imputed_rating'], ascending=[True, False])\
.merge(movie_titles, how='left', on='movie_id', validate='m:1')\
.rename(columns={'name': 'recommended_movie_name',
                 'year': 'year_of_release'})
print('For each user, pick the top 5 movies that they have seen (to be used to merge in).')
recommendations.head(10)
agg_rec = recommendations.groupby(['recommended_movie_name', 'year_of_release'])['cust_id'].nunique()\
                        .sort_values(ascending=True)
rb_palette = [(x/10.0, x/100.0, x/40.0) for x in range(len(agg_rec.tail(10)))] 
# <-- gradient rgb   (x/10.0, x/20.0, 0.75)
ax = agg_rec.tail(10).plot(kind='barh', x='cust_id', color=rb_palette)
ax.set_title('Most recommended movies to Users')
ax.set_ylabel('')
ax.set_xlabel('Number of times movie made it to Users Top 5 Recommendation')
print('Top Movie Recommendations')
userSim = mat.copy(deep=True)
corr = userSim.T.corr(min_periods=50)    # pairwise pearson correlation coefficient of columns
corr.head(5)
threshold = 0.10
print('Set the threshold similarity between users to be .1 given the distribution of corrs.')
pd.Series(np.triu(corr.values).flatten()).dropna().describe(percentiles=[x*.1 for x in range(10)]).round(2)
nearest_users = corr.stack()\
.reset_index(level=1)\
.rename(columns={
    'cust_id': 'cust_id_2',
    0: 'similarity_score'})\
.reset_index()\
.query('similarity_score > {}'.format(threshold))\
.query('cust_id != cust_id_2')\
.groupby('cust_id')\
.apply(lambda x: x.nlargest(5, columns='similarity_score'))\
.reset_index(drop=True)\
.sort_values(by=['cust_id', 'similarity_score'], ascending=[True, False])
print('For each user, get the nearest 5 users (not themselves) that are above the threshold similarity.')
nearest_users.head(10)
top5perUser = mat.stack()\
.reset_index()\
.rename(columns={0: 'rating'})\
.groupby('cust_id')\
.apply(lambda x: x.nlargest(5, columns='rating'))\
.reset_index(drop=True)\
.sort_values(by=['cust_id', 'rating'], ascending=[True, False])\
.merge(movie_titles, how='left', on='movie_id', validate='m:1')\
.rename(columns={'name': 'recommended_movie',
                 'year': 'year_of_release'})
top5perUser.head(10)
top5perUser['rank'] = top5perUser.assign(count=1).groupby('cust_id')['count'].transform('cumsum')
top5recs = top5perUser.drop(['movie_id', 'rating', 'year_of_release'], axis=1)\
.set_index(['cust_id', 'rank'])\
.unstack().reset_index()
top5recs.columns = ['_'.join([str(x) for x in col]).strip('_') for col in top5recs.columns.values]
print('For each user, get the top 5 recommended movies.')
top5recs.sample(5)
userUserRecs = nearest_users.merge(top5recs, how='left', left_on='cust_id_2', right_on='cust_id', suffixes=['', '_'])\
.drop('cust_id_', axis=1)
print('For each customer, merge in the similar users recommended movies')
userUserRecs.head(10)
seenMovie = mat.stack()\
.reset_index()\
.rename(columns={0: 'user_rating'})\
.assign(customer_seen_movie_flag = 1)
seenMovie.head()
threshold = .6
weightedRatings = mat.stack()\
.reset_index()\
.rename(columns={0: 'rating'})\
.merge(nearest_users, how='right', left_on='cust_id', right_on='cust_id_2', suffixes=['_', ''])\
.drop('cust_id_', axis=1)\
.query('similarity_score > {}'.format(threshold))\
.query('cust_id != cust_id_2')\
.assign(user_rating_weighted_by_similarity = lambda x: (x.rating * x.similarity_score))\
.groupby(['cust_id', 'movie_id'], as_index=False)[['user_rating_weighted_by_similarity', 'similarity_score']].sum()\
.assign(prediction = lambda x: (x.user_rating_weighted_by_similarity / x.similarity_score))\
.sort_values(by=['cust_id', 'prediction'], ascending=[True, False])\
.merge(movie_titles, how='left', on='movie_id', validate='m:1')\
.rename(columns={'name': 'recommended_movie',
                 'year': 'year_of_release'})\
.merge(seenMovie, how='left', on=['cust_id', 'movie_id'])
weightedRatings.sample(10)
print('Threshold pearsons score: {}.'.format(threshold))
print('Increasing the threshold will decrease the number of customers we can provide \n recommendations for but increases the quality of the recommendation.')
print('------------')
print('Recommendations available for {} out of {} users.'.format(weightedRatings.cust_id.nunique(), mat.shape[0]))
weightedRatings.groupby('cust_id', as_index=False).agg({'recommended_movie': 'nunique', 'similarity_score': {'min', 'mean', 'max'}})\
.rename(columns={'recommended_movie': 'number_of_recommendations'}).sample(10)
print('Subset to movies customers havent seen.')
weightedRatings.loc[(weightedRatings.customer_seen_movie_flag != 1) & (weightedRatings.prediction > 4)].sample(10)
sub = weightedRatings.loc[(weightedRatings.customer_seen_movie_flag == 1)]
sample_size = round(.2 * sub.shape[0])
sample = sub.sample(sample_size)
from scipy.stats.stats import pearsonr
stats = pearsonr(sample.user_rating, sample.prediction)
ax = sns.stripplot(x="user_rating", y="prediction", data=sample, jitter=True, color='red', size=3)
ax.set_title('User Ratings to Predicted Rating')
ax.set_ylabel('Prediction')
ax.set_xlabel('Current User Rating')
print('Lets use the data where the customer has seen the movie to evaluate these results.')
print('---------------')
print('User ratings seem to be overall trending positively with predicted value.')
print('Pearsons Correlation: {}, P-value: {}'.format(stats[0], stats[1]))