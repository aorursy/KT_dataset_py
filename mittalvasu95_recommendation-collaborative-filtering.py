# import libraties

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

from sklearn.model_selection import train_test_split



# Data display coustomization

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)



# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
# Reading movies file



movies = pd.read_csv('/kaggle/input/movielens-latest-small/movies.csv', encoding='latin-1')

movies.head()
print('Shape:', movies.shape)

print('Movie ids:', movies.movieId.nunique())

print('Titles:', movies.title.nunique())
# Reading ratings file



ratings = pd.read_csv('/kaggle/input/movielens-latest-small/ratings.csv', encoding='latin-1')

ratings.head()
print('Shape:', ratings.shape)

print('Movie ids:', ratings.movieId.nunique())

print('Number of users:', ratings.userId.nunique())
# These are the movies that have been stored with two different ids



movies.title.value_counts().sort_values(ascending=False).head(5)
# getting the ids of a movie

movies[movies['title'] == 'Saturn 3 (1980)']
# checking those ids in 'rating' dataframe and count which id is most watched

ratings[(ratings['movieId'] == 2851) | (ratings['movieId'] == 168358)]['movieId'].value_counts()
# deleting the id who is less watched

movies = movies[movies['movieId'] != 168358]
# getting the ids of a movie

movies[movies['title'] == 'Confessions of a Dangerous Mind (2002)']
# checking those ids in 'rating' dataframe and count which id is most watched

ratings[(ratings['movieId'] == 6003) | (ratings['movieId'] == 144606)]['movieId'].value_counts()
# deleting the id who is less watched

movies = movies[movies['movieId'] != 144606]
# getting the ids of a movie

movies[movies['title'] == 'Emma (1996)']
# checking those ids in 'rating' dataframe and count which id is most watched

ratings[(ratings['movieId'] == 838) | (ratings['movieId'] == 26958)]['movieId'].value_counts()
# deleting the id who is less watched

movies = movies[movies['movieId'] != 26958]
# getting the ids of a movie

movies[movies['title'] == 'War of the Worlds (2005)']
# checking those ids in 'rating' dataframe and count which id is most watched

ratings[(ratings['movieId'] == 34048) | (ratings['movieId'] == 64997)]['movieId'].value_counts()
# deleting the id who is less watched

movies = movies[movies['movieId'] != 64997]
# getting the ids of a movie

movies[movies['title'] == 'Eros (2004)']
# checking those ids in 'rating' dataframe and count which id is most watched

ratings[(ratings['movieId'] == 32600) | (ratings['movieId'] == 147002)]['movieId'].value_counts()
# deleting the id who is less watched

movies = movies[movies['movieId'] != 147002]
movies_ratings = pd.merge(movies, ratings, on='movieId')

movies_ratings.head()
movies_ratings.shape
# dropping 'timestamp' column

movies_ratings = movies_ratings[['userId','movieId', 'title', 'genres', 'rating']]



# sort the dataframe according to 'userId' and then 'movieId'

movies_ratings.sort_values(['userId','movieId'], inplace=True)



# resetting the index

movies_ratings.reset_index(drop=True, inplace=True)



# top 10 rows

movies_ratings.head(10)
# number of customer ids

movies_ratings.userId.nunique()
# number of movie ids

movies_ratings.movieId.nunique()
# number of movie titles

movies_ratings.title.nunique()
# removing the extra whitespaces(if any) from the column 'title' and 'genres'

movies_ratings['title'] = movies_ratings['title'].str.strip()

movies_ratings['genres'] = movies_ratings['genres'].str.strip()



# extracting the 'year'

movies_ratings['year'] = movies_ratings['title'].str[-5:-1]
movies_ratings.year.unique()
movies_ratings['year'] = movies_ratings['year'].replace('irro',2011)

movies_ratings['year'] = movies_ratings['year'].replace('atso',2011)

movies_ratings['year'] = movies_ratings['year'].replace(' Bab',2017)

movies_ratings['year'] = movies_ratings['year'].replace('ron ',2017)

movies_ratings['year'] = movies_ratings['year'].replace('r On',2018)

movies_ratings['year'] = movies_ratings['year'].replace('lon ',1994)

movies_ratings['year'] = movies_ratings['year'].replace('imal',2016)

movies_ratings['year'] = movies_ratings['year'].replace('osmo',2019)

movies_ratings['year'] = movies_ratings['year'].replace('he O',2016)

movies_ratings['year'] = movies_ratings['year'].replace(' Roa',2015)

movies_ratings['year'] = movies_ratings['year'].replace('ligh',2016)

movies_ratings['year'] = movies_ratings['year'].replace('erso',2016)
# movieIds where genre is missing



movies_ratings[movies_ratings['genres']=='(no genres listed)'].drop_duplicates('movieId')['movieId'].values
movies_ratings.loc[movies_ratings['movieId']==122896,"genres"] = 'Adventure|Action|Fantasy'

movies_ratings.loc[movies_ratings['movieId']==114335,"genres"] = 'Fantasy'

movies_ratings.loc[movies_ratings['movieId']==174403,"genres"] = 'Documentary|Biography'

movies_ratings.loc[movies_ratings['movieId']==172591,"genres"] = 'Crime|Drama|Thriller'

movies_ratings.loc[movies_ratings['movieId']==176601,"genres"] = 'Sci-Fi|Fantasy'

movies_ratings.loc[movies_ratings['movieId']==155589,"genres"] = 'Comedy'

movies_ratings.loc[movies_ratings['movieId']==147250,"genres"] = 'Crime|Mystery|Romance'

movies_ratings.loc[movies_ratings['movieId']==171749,"genres"] = 'Animation|Crime|Drama'

movies_ratings.loc[movies_ratings['movieId']==173535,"genres"] = 'Crime|Drama|Mystery'

movies_ratings.loc[movies_ratings['movieId']==134861,"genres"] = 'Comedy'

movies_ratings.loc[movies_ratings['movieId']==159161,"genres"] = 'Comedy'

movies_ratings.loc[movies_ratings['movieId']==171631,"genres"] = 'Documentary|Comedy'

movies_ratings.loc[movies_ratings['movieId']==171891,"genres"] = 'Documentary'

movies_ratings.loc[movies_ratings['movieId']==142456,"genres"] = 'Comedy|Fantasy'

movies_ratings.loc[movies_ratings['movieId']==181413,"genres"] = 'Documentary'

movies_ratings.loc[movies_ratings['movieId']==159779,"genres"] = 'Comedy|Fantasy'

movies_ratings.loc[movies_ratings['movieId']==169034,"genres"] = 'Musical'

movies_ratings.loc[movies_ratings['movieId']==171495,"genres"] = 'Sci-Fi'

movies_ratings.loc[movies_ratings['movieId']==172497,"genres"] = 'Action|Sci-Fi'

movies_ratings.loc[movies_ratings['movieId']==166024,"genres"] = 'Drama|Music'

movies_ratings.loc[movies_ratings['movieId']==167570,"genres"] = 'Drama|Fantasy|Mystery'

movies_ratings.loc[movies_ratings['movieId']==129250,"genres"] = 'Comedy'

movies_ratings.loc[movies_ratings['movieId']==143410,"genres"] = 'Action|Drama|War'

movies_ratings.loc[movies_ratings['movieId']==149330,"genres"] = 'Animation|Sci-Fi'

movies_ratings.loc[movies_ratings['movieId']==182727,"genres"] = 'Musical'

movies_ratings.loc[movies_ratings['movieId']==152037,"genres"] = 'Romance|Musical'

movies_ratings.loc[movies_ratings['movieId']==165489,"genres"] = 'Drama|Animation|History'

movies_ratings.loc[movies_ratings['movieId']==141866,"genres"] = 'Horror|Music|Thriller'

movies_ratings.loc[movies_ratings['movieId']==122888,"genres"] = 'Action|Adventure|Drama'

movies_ratings.loc[movies_ratings['movieId']==156605,"genres"] = 'Comedy|Drama|Romance'

movies_ratings.loc[movies_ratings['movieId']==141131,"genres"] = 'Action|Mystery|Sci-Fi'

movies_ratings.loc[movies_ratings['movieId']==181719,"genres"] = 'Biography|Drama'

movies_ratings.loc[movies_ratings['movieId']==132084,"genres"] = 'Drama|Romance'

movies_ratings.loc[movies_ratings['movieId']==161008,"genres"] = 'Drama|Music|Romance'
# replacing 'musical' with 'music' as both have same meaning

movies_ratings['genres'] = movies_ratings['genres'].str.replace('Musical','Music')
# converting string to int

movies_ratings['year'] = movies_ratings['year'].astype(int)
movies_ratings.info()
movies_ratings.head()
# store the column in different dataframe

genre_df = movies_ratings[['genres']]



# splitting the columns

genre_df = genre_df['genres'].str.split('|', expand=True)



genre_df.head()
# changing the name of the columns

genre_df.rename(columns={0:'G1',1:'G2',2:'G3',3:'G4',4:'G5',5:'G6',6:'G7',7:'G8',8:'G9',9:'G10'}, inplace=True)
# create a function that return distinct genres from whole dataframe



def genre_name(dataframe):

    df = dataframe.copy()

    col = df.columns

    u = set()

    for i in col:

        s = set(df[i].value_counts().index)

        u = u.union(s)

    return(u)
# names of distinct genres (21 genres)

g = genre_name(genre_df)

g
# making columns of each of the genes with value either 1 or 0 in original dataframe 



for genre in g:

    movies_ratings[genre] = movies_ratings['genres'].apply(lambda x: 1 if genre in x else 0)
movies_ratings.head()
plt.figure(figsize=(10,5))

plt.hist(movies_ratings['rating'],bins=10, color='pink', alpha=0.7)

plt.xlabel('rating',size=12)

plt.xlim(0.5,5)

plt.ylim(0,30000)

plt.vlines(x=3.5, ymin=0, ymax=30000, color='red', label='Mean rating')

plt.ylabel('')

plt.title('count plot of ratings',size=18, color='red')

plt.legend()

plt.show()
genres_count = movies_ratings.iloc[:,6:].sum(axis=0).reset_index().rename(columns={'index':'genre',0:'count'})

genres_count.sort_values('count',ascending=False, inplace=True)



plt.figure(figsize=(15,5))

sns.barplot(x = genres_count['genre'], y=genres_count['count'], color='lightgreen')

plt.xticks(rotation=45)

plt.xlabel('Genres', size=12)

plt.ylabel('')

plt.title('Count plot of genres', size=18, color='green')

plt.show()
mr = movies_ratings.groupby('title')['title'].count().sort_values(ascending=False).head(15)



plt.figure(figsize=(10,5))

sns.barplot(y = mr.index, x=mr.values, color='skyblue')

plt.ylabel('')

plt.title('15 Most watched Movies', size=18, color='blue')

plt.show()
user = movies_ratings.groupby('userId')['title'].count().sort_values(ascending=False).head(20)



plt.figure(figsize=(15,5))

user.plot(kind="bar", color="orange", alpha=0.5)

plt.title("Top 20 users according to watched history", size=18, color='orange')

plt.xlabel('User Id', size=12)

plt.xticks(rotation=0)

plt.show()
def best_movie(dataframe):

    """

    This function will return a dataframe in which there are 3 columns. The first column is year.

    The second column is number of movies released in that year. (according to data we have)

    Third column is the most watched movie of that year. (in the given data)

    It only takes one argument which is data.

    """

    df = dataframe.copy()

    movieid = df.year.unique()

    year = list()

    nMovies= list()

    mostWatched = list()

    for i in movieid:

        year.append(i)

        nMovies.append(df[df['year']==i]['title'].nunique())

        mostWatched.append(df[df['year']==i]['title'].value_counts().index[0])

    

    df1 = pd.DataFrame({'year':year,'nMoviesReleased':nMovies, 'mostWatchedMovie':mostWatched})

    df1.sort_values('year', inplace=True)

    return(df1)
# calling the function and reading its top 10 rows

yearWiseBestMovie = best_movie(movies_ratings)

yearWiseBestMovie.head(10)
train, test = train_test_split(ratings, test_size=0.30, random_state=31)
print(train.shape)

print(test.shape)
dummy_train = train.copy()

dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)



dummy_test = test.copy()

dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x>=1 else 0)
# The movies not rated by user is marked as 1 for prediction. 

dummy_train = dummy_train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(1)



# The movies not rated by user is marked as 0 for evaluation. 

dummy_test = dummy_test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(0)
# 1 means not watched by user and 0 means watched by user

dummy_train.head()
dummy_train.shape
# 0 means not watched by user and 1 means watched by user

dummy_test.head()
dummy_test.shape
# pivot ratings into movie features

df_movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).fillna(0)
df_movie_features.head()
df_movie_features.shape
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix using 'cosine' measure



user_correlation = 1 - pairwise_distances(df_movie_features, metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation.shape
movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating')
movie_features.head()
movie_features.shape
mean = np.nanmean(movie_features, axis=1) # nanmean calculate the mean excluding NaN values

print(mean.shape)
# finally subtracting each user mean rating from its own values



df_subtracted = (movie_features.T - mean).T

df_subtracted.head()
df_subtracted.shape
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')

user_correlation[np.isnan(user_correlation)] = 0

print(user_correlation)
user_correlation.shape
user_correlation[user_correlation<0]=0

user_correlation
# These are the scores of all the movies by all users 



user_predicted_ratings = np.dot(user_correlation, movie_features.fillna(0)) # 610x610 . 610x8536 = 610x8536

user_predicted_ratings
user_predicted_ratings.shape
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)   # 610x8536 x 610x8536 = 610x8536

user_final_rating.head()
def top_10_movies_for_user(i):

    user_i = user_final_rating.iloc[i].to_frame()

    user_i.reset_index(inplace=True)

    user_i.rename(columns= {'movieId':'movieId', i+1:'ratings'}, inplace=True)

    user_join_i = pd.merge(user_i, movies, on='movieId')

    return user_join_i.sort_values(by=["ratings"], ascending=False)[0:10]
top_10_movies_for_user(540)
test_movie_features = test.pivot(

    index='userId',

    columns='movieId',

    values='rating')



test_movie_features.head()
mean = np.nanmean(test_movie_features, axis=1)

test_df_subtracted = (test_movie_features.T-mean).T
# User Similarity Matrix

test_user_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_user_correlation[np.isnan(test_user_correlation)] = 0

print(test_user_correlation)
test_user_correlation.shape
test_user_correlation[test_user_correlation<0]=0

test_user_predicted_ratings = np.dot(test_user_correlation, test_movie_features.fillna(0))

test_user_predicted_ratings
test_user_predicted_ratings.shape
test_user_final_rating = np.multiply(test_user_predicted_ratings, dummy_test)
test_user_final_rating.head()
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_user_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))



print(y)
test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating')
# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)
movie_features = train.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).T



movie_features.head()
mean = np.nanmean(movie_features, axis=1) # nanmean calculate the mean excluding NaN values

print(mean.shape)
# finally subtracting each user mean rating from its own values



df_subtracted = (movie_features.T-mean).T

df_subtracted.head()
df_subtracted.shape
from sklearn.metrics.pairwise import pairwise_distances



# User Similarity Matrix

item_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')

item_correlation[np.isnan(item_correlation)] = 0

print(item_correlation)
item_correlation.shape
item_correlation[item_correlation<0]=0

item_correlation
item_predicted_ratings = np.dot((movie_features.fillna(0).T), item_correlation)

item_predicted_ratings
item_predicted_ratings.shape
dummy_train.shape
item_final_rating = np.multiply(item_predicted_ratings, dummy_train)

item_final_rating.head()
def top_10_movies_for_user(i):

    user_i = item_final_rating.iloc[i].to_frame()

    user_i.reset_index(inplace=True)

    user_i.rename(columns= {'movieId':'movieId', i+1:'ratings'},inplace=True)

    user_join_i = pd.merge(user_i,movies,on='movieId')

    return user_join_i.sort_values(by=["ratings"],ascending=False)[0:10]
top_10_movies_for_user(540)
test_movie_features = test.pivot(

    index='userId',

    columns='movieId',

    values='rating'

).T
mean = np.nanmean(test_movie_features, axis=1)

test_df_subtracted = (test_movie_features.T-mean).T
test_item_correlation = 1 - pairwise_distances(test_df_subtracted.fillna(0), metric='cosine')

test_item_correlation[np.isnan(test_item_correlation)] = 0

test_item_correlation[test_item_correlation<0]=0
test_item_correlation.shape
test_movie_features.shape
test_item_predicted_ratings = (np.dot(test_item_correlation, test_movie_features.fillna(0))).T

test_item_final_rating = np.multiply(test_item_predicted_ratings,dummy_test)

test_item_final_rating.head()
test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating')
from sklearn.preprocessing import MinMaxScaler

from numpy import *



X  = test_item_final_rating.copy() 

X = X[X>0]



scaler = MinMaxScaler(feature_range=(1, 5))

print(scaler.fit(X))

y = (scaler.transform(X))



test_ = test.pivot(

    index='userId',

    columns='movieId',

    values='rating')



# Finding total non-NaN value

total_non_nan = np.count_nonzero(~np.isnan(y))
rmse = (sum(sum((test_ - y )**2))/total_non_nan)**0.5

print(rmse)