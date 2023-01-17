import numpy as np
import pandas as pd
#Import movie ratings
data_ratings = pd.read_csv('../input/movie-lens-small-latest-dataset/ratings.csv')
data_ratings.head()
#Import movie data
data_movies = pd.read_csv('../input/movie-lens-small-latest-dataset/movies.csv')
data_movies.head()
#Merge both the datasets
movie_ratings = pd.merge(data_movies, data_ratings, on = 'movieId')
print(movie_ratings.shape)
movie_ratings.head()
#Groupby all movie titles together and find their mean ratings
movie_ratings.groupby('title')['rating'].mean().head()
#Sort movies based on ratings from highest to lowest
movie_ratings.groupby('title')['rating'].mean().sort_values(ascending = False)
#Recommend top n popular movies
n = 10

movie_ratings.groupby('title')['rating'].mean().sort_values(ascending = False).head(n)
movie_ratings['title'].value_counts()
#movie_ratings.groupby('title')['rating'].count().sort_values(ascending = False).head() either of the 2 gives same output
#First create a DataFrame
data = pd.DataFrame(movie_ratings.groupby('title')['rating'].mean())
data['rating_counts'] = pd.DataFrame(movie_ratings['title'].value_counts())
#data['rating_counts'] = pd.DataFrame(movie_ratings.groupby('title')['rating'].count()) #either of the 2 codes
data.head()
#load packages
from math import *

#Creating 2 functions, square root and cosine similarity just like the formula

def square_rooted(x):
    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator/ float(denominator),3)

print(cosine_similarity([3,45,7,2],[2,54,13,15]))