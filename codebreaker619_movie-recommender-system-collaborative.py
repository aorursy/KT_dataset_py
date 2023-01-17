# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
credits = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv")
movies = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")

credits.head()
credits.describe()
movies.head()
movies.describe()
print(credits.shape)
print(movies.shape)
credits.columns = ['id','title','cast','crew']
movies = movies.merge(credits, on="id")
movies.head()
movies.shape
movies_cleaned = movies.drop(columns = ['homepage', 'title_x', 'title_y', 'status', 'spoken_languages'])
movies_cleaned.head()
# Changing 'genres' column from json to string
movies_cleaned['genres'] = movies_cleaned['genres'].apply(json.loads)
for index,i in zip(movies_cleaned.index, movies_cleaned['genres']):
    l1 = []
    for j in range(len(i)):
        l1.append((i[j]['name']))     # "name" contains => name of the genre
    movies_cleaned.loc[index, 'genres'] = str(l1)
    
# Changing 'keywords' column from json to string
movies_cleaned['keywords'] = movies_cleaned['keywords'].apply(json.loads)
for index,i in zip(movies_cleaned.index, movies_cleaned['keywords']):
    l1 = []
    for j in range(len(i)):
        l1.append((i[j]['name']))     # "name" contains => name of the keyword
    movies_cleaned.loc[index, 'keywords'] = str(l1)
    
# Changing 'production_companies' column from json to string
movies_cleaned['production_companies'] = movies_cleaned['production_companies'].apply(json.loads)
for index,i in zip(movies_cleaned.index, movies_cleaned['production_companies']):
    l1 = []
    for j in range(len(i)):
        l1.append((i[j]['name']))     # "name" contains => name of the keyword
    movies_cleaned.loc[index, 'production_companies'] = str(l1)
    
# Changing 'production_companies' column from json to string
movies_cleaned['production_countries'] = movies_cleaned['production_countries'].apply(json.loads)
for index,i in zip(movies_cleaned.index, movies_cleaned['production_countries']):
    l1 = []
    for j in range(len(i)):
        l1.append((i[j]['name']))     # "name" contains => name of the keyword
    movies_cleaned.loc[index, 'production_countries'] = str(l1)
    
# Changing 'cast' column from json to string
movies_cleaned['cast'] = movies_cleaned['cast'].apply(json.loads)
for index,i in zip(movies_cleaned.index, movies_cleaned['cast']):
    l1 = []
    for j in range(len(i)):
        l1.append((i[j]['name']))     # "name" contains => name of the keyword
    movies_cleaned.loc[index, 'cast'] = str(l1)


movies_cleaned['crew']=movies_cleaned['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
movies_cleaned['crew']=movies_cleaned['crew'].apply(director)
movies_cleaned.rename(columns={'crew':'director'},inplace=True)

movies_cleaned.head()
movies_cleaned['genres']=movies_cleaned['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies_cleaned['genres']=movies_cleaned['genres'].str.split(',')

movies_cleaned['keywords']=movies_cleaned['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies_cleaned['keywords']=movies_cleaned['keywords'].str.split(',')

movies_cleaned['production_companies']=movies_cleaned['production_companies'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies_cleaned['production_companies']=movies_cleaned['production_companies'].str.split(',')

movies_cleaned['production_countries']=movies_cleaned['production_countries'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies_cleaned['production_countries']=movies_cleaned['production_countries'].str.split(',')

movies_cleaned['cast']=movies_cleaned['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies_cleaned['cast']=movies_cleaned['cast'].str.split(',')

movies_cleaned.head()
movies_cleaned.info()
v = movies_cleaned['vote_count']
R = movies_cleaned['vote_average']
C = movies_cleaned['vote_average'].mean()
m = movies_cleaned['vote_count'].quantile(0.70)    # Movies > 70th percentile votes
movies_cleaned['weighted_avg'] = ((R*v)+(C*m))/(v+m)
movies_cleaned.head()
sorted_ranking = movies_cleaned.sort_values('weighted_avg', ascending=False)
sorted_ranking
sorted_ranking[['original_title', 'vote_count', 'vote_average', 'weighted_avg', 'popularity']].head(20)
weight_avg = sorted_ranking.sort_values('weighted_avg', ascending=False)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x = weight_avg['weighted_avg'].head(10), y=weight_avg['original_title'].head(10), data=weight_avg)
plt.xlim(4, 10)
plt.title("Best Movies by Votes", weight="bold")
plt.xlabel("Weighted Average Score", weight="bold")
plt.ylabel("Movie Title", weight="bold")
plt.show()
popularity = sorted_ranking.sort_values('popularity', ascending=False)
popularity.head(10)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x = popularity['popularity'].head(10), y=popularity['original_title'].head(10), data=popularity)
plt.title("Best Movies by Popularity", weight="bold")
plt.xlabel("Popularity Score", weight="bold")
plt.ylabel("Movie Title", weight="bold")
plt.show()
# Scaling down the Popularity Score and Weighted Average due to difference in magnitude
scaling = MinMaxScaler()
movie_scaled = scaling.fit_transform(movies_cleaned[['weighted_avg', 'popularity']])
movie_normalized = pd.DataFrame(movie_scaled, columns=['weighted_avg', 'popularity'])
movie_normalized.head()
movies_cleaned[['normalized_weighted_avg', 'normalized_popularity']] = movie_normalized
movies_cleaned.head(20)
movies_cleaned['score'] = movies_cleaned['normalized_weighted_avg'] * 0.5 + movies_cleaned['normalized_popularity'] * 0.5
movies_cleaned = movies_cleaned.sort_values(['score'], ascending=False)
movies_cleaned[['original_title','normalized_weighted_avg', 'normalized_popularity', 'score']].head(10)
movies_score = movies_cleaned.sort_values('score', ascending=False)

plt.figure(figsize=(16, 6))
ax = sns.barplot(x = movies_score['score'].head(10), y=movies_score['original_title'].head(10), data=movies_score)
plt.title("Best Rated and Most Popular Movies", weight="bold")
plt.xlabel("Score", weight="bold")
plt.ylabel("Movie Titles", weight="bold")
plt.show()
df = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")
df.head()
df.shape
# Using smaller amount of data => otherwise pandas gives error => pivot_table on large data does not work, int32 overflow
df = df[:100003]
df.shape
titles = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
titles.head()
# Merge the ratings and movies dataframe
df = pd.merge(df, titles, on="movieId")
df.head()
# Sort the rating from highest to lowest based on the rating value
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# Sort based on the count of number of ratings given to the movies
df.groupby('title')['rating'].count().sort_values(ascending=False).head()
# Storing the mean values of the ratings for each movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()
ratings['num_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
# Plot histogram wrt number of ratings
plt.figure(figsize=(10,4))
ratings['num_of_ratings'].hist(bins=70)
plt.figure(figsize=(10,4))
ratings['rating'].hist(bins=70)    # Follow normal Gaussian Distribution with some outliers
sns.jointplot(x='rating', y='num_of_ratings', data=ratings, alpha=0.5)
ratings.sort_values('num_of_ratings', ascending=False).head(10)
moviemat = df.pivot_table(index="userId", columns="title", values='rating')
moviemat.head()
forrest_gump_user_ratings = moviemat['Forrest Gump (1994)']
shawshank_user_ratings = moviemat['Shawshank Redemption, The (1994)']

forrest_gump_user_ratings.head()
shawshank_user_ratings.head()
# Find correlations
similar_forrest_gump = moviemat.corrwith(forrest_gump_user_ratings)
similar_shawshank = moviemat.corrwith(shawshank_user_ratings)
similar_forrest_gump.head()
similar_shawshank.head()
# Drop 'NaN' values and convert the correlations to a dataframe
# Higher Correlation => first recommendation
# Max Correlation = 1
corr_forrest_gump = pd.DataFrame(similar_forrest_gump, columns=['Correlation'])
corr_forrest_gump.dropna(inplace=True)
corr_forrest_gump.head()
corr_forrest_gump.shape
corr_forrest_gump.sort_values('Correlation', ascending=False).head(20)
corr_shawshank = pd.DataFrame(similar_shawshank, columns=['Correlation'])
corr_shawshank.dropna(inplace=True)
corr_shawshank.head()
corr_shawshank.shape
corr_shawshank.sort_values('Correlation', ascending=False).head(20)
corr_forrest_gump = corr_forrest_gump.join(ratings['num_of_ratings'])
corr_forrest_gump.head(10)
corr_shawshank = corr_shawshank.join(ratings['num_of_ratings'])
corr_shawshank.head(10)
# Considering correlations where number of ratings>100
corr_forrest_gump[corr_forrest_gump['num_of_ratings']>100].sort_values('Correlation', ascending=False).head(10)
corr_shawshank[corr_shawshank['num_of_ratings']>100].sort_values('Correlation', ascending=False).head(10)
movies = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv", usecols=['movieId','title'],
                    dtype={'movieId':'int32', 'title': 'str'})
ratings = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv", usecols=['userId', 'movieId', 'rating'],
                     dtype={'userId':'int32', 'movieId':'int32', 'rating':'float32'})
movies.head()
ratings.head()
df = pd.merge(movies, ratings, on="movieId")
df.head()
# Count Ratings for each and every movie
ratings = df.dropna(axis=0, subset = ['title'])    # Drop all 'NaN' values
movie_rating_Count = (ratings.groupby(by=['title'])['rating'].count().reset_index().
                     rename(columns = {'rating':'TotalRatingCount'})[['title', 'TotalRatingCount']])

movie_rating_Count.head(10)
movie_rating_Count.describe()
# Merging the rating counts with the ratings
ratings = ratings.merge(movie_rating_Count, left_on='title', right_on='title', how='left')
# left_on => on left dataframe which column considered, right_on => on right dataframe which column considered
ratings.head()
plt.figure(figsize=(10,4))
ratings['TotalRatingCount'].hist(bins=70)
popularity_threshold = 10000
rating_popular_movie = ratings.query('TotalRatingCount >= @popularity_threshold')
rating_popular_movie.head()
rating_popular_movie.shape
s = set(rating_popular_movie['title'])
s
# Create a Pivot Table
features = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
features.head()
# Convert the pivot_table into an array matrix
from scipy.sparse import csr_matrix
features_matrix = csr_matrix(features.values)    # All info of pivot table converted into an array
features_matrix
from sklearn.neighbors import NearestNeighbors   # Not KNearestNeighbors, NearestNeighbors => Unsupervised Algo
model = NearestNeighbors(metric = "cosine", algorithm="brute")
model.fit(features_matrix)    # p=2 => Euclidean Distance Parameter
features.shape
# Taking a new movie at random
query_index = np.random.choice(features.shape[0])    # Collect 1 record
print(query_index)
# Find similar movies(nearer to the selected movie) using kneighbors
distances, indices = model.kneighbors(features.iloc[query_index,:].values.reshape(1, -1), n_neighbors=6)
# n_neighbors = 6 => will include the movie itself => We will be getting 5 other movie recommendations
# distances = 0 => Same movie itself
distances
indices
# Print top 5 movie name recommendations for movie along with the distances from original movie
for i in range(0, len(distances.flatten())):     # Convert 'distances' array into 1-D array
    if(i==0):
        print("Recommendations for {0}:\n".format(features.index[query_index]))    # 1st recommendation => same movie itself
    else:
        print("{0}: {1}, with distance of {2}:".format(i, features.index[indices.flatten()[i]], distances.flatten()[i]))