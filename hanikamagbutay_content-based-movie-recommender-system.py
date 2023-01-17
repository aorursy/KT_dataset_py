import pandas as pd

from math import sqrt

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
movies_df = pd.read_csv('/kaggle/input/movielens/movies.dat', sep = '::', engine='python', names = ['MovieID','Title','Genre'])

ratings_df = pd.read_csv('/kaggle/input/movielens/ratings.dat', sep = '::', engine='python', names = ['UserID','MovieID','Rating','Timestamp'])
movies_df.head()
# Remove year from the Title column and store it in a new column

movies_df['Year'] = movies_df.Title.str.extract('(\(\d\d\d\d\))',expand=False)

movies_df['Year'] = movies_df.Year.str.extract('(\d\d\d\d)',expand=False)

movies_df['Title'] = movies_df.Title.str.replace('(\(\d\d\d\d\))', '')

movies_df['Title'] = movies_df['Title'].apply(lambda x: x.strip())



# Split the values in the Genre column

movies_df['Genre'] = movies_df.Genre.str.split('|')



movies_df.head()
movies_with_genres = movies_df.copy()



for index, row in movies_df.iterrows():

    for genre in row['Genre']:

        movies_with_genres.at[index, genre] = 1



movies_with_genres = movies_with_genres.fillna(0)

movies_with_genres.head()
ratings_df.head()
ratings_df = ratings_df.drop('Timestamp', 1)

ratings_df.head()
import random

user = random.choice(ratings_df['UserID'])

print("User ID: ", user)
user_input = ratings_df[ratings_df['UserID'] == user].reset_index(drop=True)

user_input
input_movies = movies_df[movies_df['MovieID'].isin(user_input['MovieID'].tolist())]

user_input = pd.merge(input_movies, user_input)

user_input = user_input.drop('Genre', 1).drop('Year', 1).drop('UserID', 1)

user_input
user_movies = movies_with_genres[movies_with_genres['MovieID'].isin(user_input['MovieID'].tolist())]

user_movies
user_movies = user_movies.reset_index(drop=True)

user_genres = user_movies.drop('MovieID', 1).drop('Title', 1).drop('Genre', 1).drop('Year', 1)

user_genres.head()
user_profile = user_genres.transpose().dot(user_input['Rating'])

user_profile
genre_table = movies_with_genres.set_index(movies_with_genres['MovieID'])

genre_table = genre_table.drop('MovieID', 1).drop('Title', 1).drop('Genre', 1).drop('Year', 1)

genre_table.head()
recommendations_df = ((genre_table*user_profile).sum(axis=1))/(user_profile.sum())

recommendations_df.head()
recommendations_df = recommendations_df.sort_values(ascending=False)

recommendations_df.head()
df = movies_df.loc[~movies_df['MovieID'].isin(user_input.MovieID)].reset_index(drop=True)

df.loc[df['MovieID'].isin(recommendations_df.head(20).keys())].reset_index(drop=True)