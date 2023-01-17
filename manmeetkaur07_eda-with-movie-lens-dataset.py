# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
column_names = ['user_id', 'item_id', 'rating','timestamp']

df = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.data', sep='\t', names=column_names)
df.head(n=5)
df.shape
# To check the number of unique users

df['user_id'].nunique()
# Checking the number of unique movies

df['item_id'].nunique()
# Retrievin the movie titles

movies_title = pd.read_csv('../input/movielens-100k-dataset/ml-100k/u.item', sep='\|', header=None)
movies_title.shape
movies_title = movies_title[[0,1]]
movies_title.columns = ['item_id', 'title']
movies_title.head(n=5)
# Merging the earlier dataframe on the basis of item_id with the movies_title

df = pd.merge(df, movies_title, on='item_id')
# Validating the merge...

df.tail()
# Finding the average rating of a movie

df.groupby('title').mean()['rating'].sort_values(ascending=False)
# How many times a movie has been watched (arranged in descending order)

df.groupby('title').count()['rating'].sort_values(ascending=False)
# Create a DF of the ratings for movies

ratings_df = pd.DataFrame(df.groupby('title').mean()['rating'])
# Adding the number of ratings column to the df created earlier

ratings_df['number of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])
ratings_df.head()
ratings_df.sort_values(by='rating', ascending=False)
# Checking the distribution of number of ratings vs appearances

plt.figure(figsize=(10,6))

plt.hist(ratings_df['number of ratings'], bins=70)

plt.xlabel('No. of user ratings')

plt.ylabel('The appearances for every rating')

plt.title('Distribution of no. of ratings')

plt.show()
# Distribution of ratings

plt.hist(ratings_df['rating'],bins=70)

plt.xlabel('Avg. rating')

plt.ylabel('No. of rating')

plt.show()



# Aha, a normal distribution spotted !!!
sns.jointplot(x = ratings_df['rating'], y = ratings_df['number of ratings'], data = ratings_df, alpha = 0.5)
# Create a matrix as user vs movie matrix with each cell having the rating for the corresponding movie

movie_matrix = df.pivot_table(index='user_id',columns='title',values='rating')
movie_matrix
# Mostly / highly watched movies

ratings_df.sort_values('number of ratings', ascending=False).head()
# User-wise rating of a particular movie

starwars_usr_ratings = movie_matrix['Star Wars (1977)']

starwars_usr_ratings.head()
# How much correlated is Star Wars with other movie

similar_to_starwars = movie_matrix.corrwith(starwars_usr_ratings)
similar_to_starwars

# NaN means the user didn't watch both the movies
corr_of_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
# Dropping NaN values

corr_of_starwars.dropna(inplace=True)
# A couple of other movies similar to Star wars

corr_of_starwars.sort_values('Correlation', ascending=False).head(10)
corr_of_starwars = corr_of_starwars.join(ratings_df['number of ratings'])
corr_of_starwars[corr_of_starwars['number of ratings'] > 100].sort_values('Correlation', ascending=False)
# Voila, we got recommended to watch the series of Star Wars... 

# Well, this was fun...
# Ok, let's create a function and check if we can implement the same thing

def recommend_movies(movie_name):

    movie_usr_ratings = movie_matrix[movie_name]

    similar_movie = movie_matrix.corrwith(movie_usr_ratings)

    

    corr_of_movie = pd.DataFrame(similar_movie, columns=['Correlation'])

    corr_of_movie.dropna(inplace=True)

    

    corr_of_movie = corr_of_movie.join(ratings_df['number of ratings'])

    

    predictions = corr_of_movie[corr_of_movie['number of ratings'] > 100].sort_values('Correlation', ascending=False)

    

    return predictions
predictions = recommend_movies('Crash (1996)')

predictions.head()