# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.max_rows = 4000
from sklearn.metrics.pairwise import cosine_similarity
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Read data
anime_data= pd.read_csv('/kaggle/input/anime-recommendations-database/anime.csv')
rating_data=pd.read_csv('/kaggle/input/anime-recommendations-database/rating.csv')
# Checking basic information about our dataset
anime_data.head()
anime_data.describe()
# The data consists of 12294 different Anime titles, it has 12064 ratings (which means some ratings are Null)
rating_data.head()
#This data contains Iser ID, Anime Id and User rating for that anime from 0 to 10.
#Items which have a rating of -1 have not been rated yet.
rating_data.describe()
# Rating data has 7.8 Million Ratings
# replacing -1 to np.Nan in rating data
rating_data.rating.replace(-1, np.NaN,inplace=True)
# Getting count of Nulls
print('Anime Data')
for i in anime_data.columns:
    print('Null counts in the column',i,':',sum(anime_data[i].isna()))

print('\n Rating Data')
for i in rating_data.columns:
    print('Null counts in the column',i,':',sum(rating_data[i].isna()))
# replacing -1 to np.Nan in rating data
anime_data.genre.replace(np.NaN,'None_Genre',inplace=True)
anime_data.type.replace(np.NaN,'None_type',inplace=True)
anime_data.episodes.replace('Unknown',np.NaN,inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import collections
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
# On average, they have a rating of 6
sns.distplot(anime_data['rating'], hist=True, kde=True, 
             bins=10, color = 'green', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
print('Average Rating:',anime_data['rating'].mean())
# On average, they have a rating of 6
sns.distplot(anime_data['members'], hist=True, kde=True, 
             bins=10, color = 'green', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
print('Average members:',anime_data['members'].mean())
print('Max members:',anime_data['members'].max())
print('Min members:',anime_data['members'].min())
print('Median members:',anime_data['members'].median())
## Check the top 10 anime with max members
anime_data.sort_values('members',ascending=False).head(10)
## Check the bottom 10 anime titles
anime_data.sort_values('members',ascending=True).head(10)
anime_data.episodes=pd.to_numeric(anime_data.episodes, errors='coerce')
# On average, they have aroud 12 episodes per series
sns.distplot(anime_data['episodes'], hist=True, kde=True, 
             bins=10, color = 'green', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
print('Average Episode Count:',anime_data['episodes'].mean())
print('Max Episode Count:',anime_data['episodes'].max())
print('Min Episode Count:',anime_data['episodes'].min())
print('Median Episode Count:',anime_data['episodes'].median())
# Anime series with highest number of episodes
anime_data.sort_values('episodes', ascending = False).head(10)
anime_data['type'].value_counts()
# Checking rating across different type of anime
sns.catplot(x="type", y="rating", kind="box", data=anime_data)

for i in anime_data['type'].unique():
    print('Average rating for',i,'anime:',anime_data[anime_data.type==i]['rating'].mean())
anime_data['genre']=anime_data['genre'].apply(lambda x : x.split(', '))
genre_data = itertools.chain(*anime_data['genre'].values.tolist())
genre_counter = collections.Counter(genre_data)
genres = pd.DataFrame.from_dict(genre_counter,orient='index').reset_index()
genres.columns=['Genre','Counts']
genres.sort_values('Counts', ascending=False, inplace=True)

# Plot genre
f, ax = plt.subplots(figsize=(10,12))
sns.barplot(x="Counts", y="Genre", data=genres, color='#719967')
ax.set(ylabel='Genre',xlabel="Count")
# Getting rankings by genre
genre_rating = []
for i in list(genres['Genre']):
    genre_rating.append(anime_data[anime_data['genre'].str.contains(i, regex=False)]['rating'].mean())

genre_rating_dict=pd.DataFrame({'Genre': list(genres['Genre']),
  'rating': genre_rating })
genre_rating_dict.sort_values('rating', ascending=False, inplace=True)
# Plot Genre - Ratings 
f, ax = plt.subplots(figsize=(10,12))
sns.barplot(x="rating", y="Genre", data=genre_rating_dict, color='#719967')
ax.set(ylabel='Genre',xlabel="Rating")

rating_anime=rating_data.merge(anime_data[['name','genre','anime_id','type','episodes','members']],left_on='anime_id',right_on='anime_id')
## Lets look at top 10 animes which have been rated the most in the dataset
top_rated= rating_anime.groupby(['anime_id','name']).count()['user_id'].reset_index().sort_values('user_id', ascending=False)
top_rated.head(10)
sns.distplot(top_rated['user_id'], hist=True, kde=True, 
             bins=10, color = 'green', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html
def popularity_recommender(dataset,N = 0, 
                           Genre =[],
                           Type= [],
                           episodes_more_than=0,
                           popularity_weight= 0.5
                          ):    
    if len(Genre)==0:
        Genre = ['Josei', 'Thriller', 'Mystery', 'Police', 'Shounen', 'Psychological', 'Military', 'Supernatural', 'Romance', 'Shoujo Ai', 'Drama', 'School', 'Seinen', 'Harem', 'Shounen Ai', 'Super Power', 'Vampire', 'Shoujo', 'Samurai', 'Martial Arts', 'Magic', 'Action', 'Game', 'Sports', 'Historical', 'Adventure', 'Slice of Life', 'Sci-Fi', 'Space', 'Demons', 'Fantasy', 'Ecchi', 'Mecha', 'Comedy', 'Parody', 'Cars', 'Yaoi', 'Horror', 'Hentai', 'Kids', 'Yuri', 'Music', 'None_Genre', 'Dementia']
    if len(Type) ==0:
        Type = ['ONA', 'None_type', 'OVA', 'Special', 'Music', 'Movie', 'TV']

    pop_recommender_df = dataset[
    (dataset.episodes>=episodes_more_than) &
    (dataset.genre.apply(len) !=(dataset.genre.apply(set)- set(Genre)).apply(len)) &
    (dataset.type.isin(Type)) ].copy()
       
    if len(pop_recommender_df)==0:
        print('No anime found with such conditions')

    else:
        # anime rating is in a range of 1 to 10 while popularity/members is in a larger range. 
        # because we want to show good shows with higher popularity, we will scale popularity on a range of 1 to 10 
        # we shall be giving 70% weight to the popularity metric and 30% weight to show rating and calculate a new score called Popularity_quality_index
        pop_recommender_df['scaled_members']=(minmax_scale(pop_recommender_df.members))*10
        pop_recommender_df['Popularity_quality_index']=(popularity_weight* pop_recommender_df.scaled_members)+(1-popularity_weight)*pop_recommender_df.rating
        df=pop_recommender_df.sort_values('Popularity_quality_index', ascending=False).iloc[0:N]
        cols =['name','genre','type','episodes','rating','members']
        return(df[cols])

genre_list=['Action']
Type_list = ['TV']
popularity_recommender(anime_data,
                       Genre=genre_list,
                       episodes_more_than=0,
                       Type=Type_list,
                       N=20,
                       popularity_weight=0.6)
# Getting rid of NA ratings
print('Original rating data size:',rating_data.shape[0])
rating_data_clean = rating_data[rating_data.rating.notna()].reset_index(drop = True).copy()
# Getting rid of Users with less than 250 ratings
print('Rating data after removing NA Ratings:',rating_data_clean.shape[0])
# Getting rid of Anime titles with less than 250 ratings 
anime_rating_counts = pd.DataFrame(rating_data_clean.groupby('anime_id')['user_id'].nunique()).reset_index()
req_anime_ids= anime_rating_counts[anime_rating_counts['user_id']>250].anime_id
user_rating_counts=pd.DataFrame(rating_data_clean.groupby('user_id')['anime_id'].nunique()).reset_index()
req_user_ids= user_rating_counts[user_rating_counts['anime_id']>250].user_id
# Splitting data into training and test sets
rating_clean=rating_data_clean[rating_data_clean.anime_id.isin(req_anime_ids) &
                              rating_data_clean.user_id.isin(req_user_ids)].copy().reset_index(drop = True)
print('Rating_data after filtering Anime Shows and Users:',rating_clean.shape[0])
# Split into training and test
train_df, test_df = train_test_split(rating_clean,
                                   test_size=0.20,
                                   random_state=27)
anime_genre_dummies= pd.get_dummies(anime_data.genre.apply(pd.Series).stack()).sum(level=0)
anime_genre_dummies=pd.concat([anime_data, anime_genre_dummies], axis=1)

def get_episode_encoding(num_episodes):
    if(num_episodes<=13):
        return('Xsmall')
    elif (num_episodes<=50):
        return('Small')
    elif (num_episodes<=250):
        return('Medium')
    elif (num_episodes<=500):
        return('Long')
    else:
        return('Xlong')
# Engineering a few features

#Making one hot encodings for type, Number of Episodes, and if the show is a very popular show, has medium popularity or lesser known popularity
# Getting Show Type dummies
anime_genre_dummies = pd.concat([anime_genre_dummies,pd.get_dummies(anime_data.type)],axis = 1)

# Getting Episode size categories
anime_genre_dummies = pd.concat([anime_genre_dummies,pd.get_dummies(anime_genre_dummies.episodes.apply(get_episode_encoding))],axis=1)

req_cols = list(set(anime_genre_dummies.columns) -set(['anime_id','name','genre','type','episodes','rating','members']))
req_cols.sort()

# Creating Encodings
anime_genre_dummies['Encoding']=anime_genre_dummies[req_cols].values.tolist()
anime_genre_dummies
def content_based_recommender(user_id,N,movie_profile):
    # get user profile
    user_profile=rating_clean[rating_clean['user_id']==user_id].copy().reset_index(drop = True)
    user_profile = user_profile[user_profile.rating.notna()]
    cols_to_merge= list(set(movie_profile.columns)-set(['rating']))
    
    user_profile =pd.merge(user_profile[['anime_id','rating']],movie_profile[cols_to_merge]
             ,how= 'left', left_on='anime_id', right_on='anime_id')
    
    req_cols = list(set(movie_profile.columns) -set(['anime_id','name','genre','type','episodes','rating','members','Encoding']))
    req_cols.sort()
    # to generate the user profile, we are summing up the genre dummy variables that the user
    #interacted with and weighing it with the rating that the user has given to the interacted items 
    user_profile['rating_scaled'] =  minmax_scale(user_profile['rating'])
    
    # Multiplying Ratings with the genre encodings
    genre_weights=pd.DataFrame(user_profile[req_cols].multiply(user_profile['rating_scaled'],axis="index").sum()).reset_index()
    genre_weights.columns=['Genre','Weights']
    
    #Scaling the encodings so that we have encodings in 0-1 range to compare with movie encodings
    genre_weights['weights_scaled']=minmax_scale(genre_weights['Weights'])
    user_profile_weights= [list(genre_weights['weights_scaled'])]
    
    # Finding cosine similarity between user profile and movies 
    movie_profile['user_affinity']= cosine_similarity(user_profile_weights,list(movie_profile['Encoding']))[0]
    
    return(movie_profile.sort_values('user_affinity', ascending = False).reset_index().iloc[1:N][['name','genre','type','episodes','rating','members','user_affinity']])

rating_clean.user_id.unique()[1:15]
content_based_recommender(user_id =17, N= 10,movie_profile = anime_genre_dummies.copy())
import surprise
from surprise.model_selection import cross_validate
from surprise import SVD,SVDpp,NMF,NormalPredictor,KNNBaseline,KNNBasic,KNNWithMeans,KNNWithZScore,BaselineOnly,CoClustering,SlopeOne, Reader, Dataset
# we will be using the Surprise Python Package to get our recommendations.
# We will try a few different Models for implementing Collaborative filtering. 

ratings_dict = {'itemID': rating_clean.anime_id,
                'userID': rating_clean.user_id,
                'rating': rating_clean.rating}
df = pd.DataFrame(ratings_dict)
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 10))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
import time 
start = time.time()


benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), NMF()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=True, n_jobs = 1)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')  
end = time.time()
print(end - start)

from collections import defaultdict

from surprise import SVD
from surprise import Dataset


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# Train an SVD algorithm on the Anime Rating cleaned dataset.

ratings_dict = {'itemID': rating_clean.anime_id,
                'userID': rating_clean.user_id,
                'rating': rating_clean.rating}
df = pd.DataFrame(ratings_dict)
# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 10))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
from collections import defaultdict

from surprise import Dataset
from surprise import SVD
from surprise.model_selection import KFold


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=7)
# Precision and recall can then be averaged over all users
print(sum(prec for prec in precisions.values()) / len(precisions))
print(sum(rec for rec in recalls.values()) / len(recalls))
# Lets use an example user
uid_test=list(top_n.keys())
def get_recommendations(uid_profile):
    user_top_10 = rating_anime[rating_anime.user_id == uid_profile].sort_values('rating',ascending = False ).iloc[1:10]
    rated_top_10= user_top_10[['name','genre','type','episodes','rating']]
    anime_id_user= [iid for (iid, _) in top_n[uid_profile]]
    top_10_recommendations = anime_data[anime_data.anime_id.isin(anime_id_user)][['name','genre','type','episodes','rating']]
    return rated_top_10,top_10_recommendations
uid_test[3]
# Getting user ratings and recommendations for user number 38

user_rated,top_10_recommendations =  get_recommendations(uid_profile=38)
print('these are the top 10 anime titles rated by user 38')
user_rated
print('these are our recommendations for the user 38')
top_10_recommendations