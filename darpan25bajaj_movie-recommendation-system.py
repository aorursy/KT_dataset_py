import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
rating = pd.read_csv('../input/movielens-latest-small/ratings.csv')
rating.head()
movies = pd.read_csv('../input/movielens-latest-small/movies.csv')
movies.head()
# merging both the datasets on 'movieId' column

movie_rating = pd.merge(left=rating,right=movies,on='movieId')
movie_rating.head()
movie_rating.columns
movie_rating = movie_rating[['userId', 'movieId', 'title', 'genres', 'rating', 'timestamp']]
movie_rating.head()
movie_rating.info()
movie_rating.isnull().sum()
movie_rating.head(2)
# grouping the movies based on average rating

average_rating_movies = movie_rating.groupby('title')['rating'].mean().sort_values(ascending=False)
average_rating_movies.head(10)
average_rating_movies.hist(bins=20)

plt.show()
# grouping the movies based on count of users who rated the movies

count_userid = movie_rating.groupby('title')['userId'].count().sort_values(ascending=False)
count_userid
count_userid.hist()

plt.show()
for movie in average_rating_movies[average_rating_movies==5.0].index:

    print(movie,count_userid[movie])
# grouping the movie_rating based on count on userId and mean on rating

userid_rating = movie_rating.groupby('title')[['userId','rating']].agg({'userId':'count','rating':'mean'}).round(2).sort_values(by='userId',ascending=False)
userid_rating.head()
# creating pivot table to create item by item collaborative filtering

movie_rating_pivot = pd.pivot_table(index='userId',columns='title',values='rating',data=movie_rating)
movie_rating_pivot.head()
userid_rating.head(10)
# assigning ratings of movie 'Jurassic Park (1993)' to a new variable from movie_rating_pivot

jurassic_park = movie_rating_pivot['Jurassic Park (1993)'].head(10)
jurassic_park.head(10)
correlation_jurassicpark = pd.DataFrame(movie_rating_pivot.corrwith(jurassic_park))
correlation_jurassicpark.head()
correlation_jurassicpark.columns = ['Correlation']

correlation_jurassicpark.dropna(inplace=True,axis=0)
correlation_jurassicpark.sort_values(by='Correlation',ascending=True).head()
correlation_jurassicpark['Views'] = userid_rating['userId']
correlation_jurassicpark[correlation_jurassicpark['Views'] > 100].sort_values(by='Correlation',ascending=False).head(20)