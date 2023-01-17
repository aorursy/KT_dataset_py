# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/tmdb_5000_credits.csv')

df2 = pd.read_csv('../input/tmdb_5000_movies.csv')
df1.info()

df1 = df1.drop(['title'], axis = 1)

# df1.head()
df2.info()
df1.columns = ['id','cast', 'crew']

df2 = df2.merge(df1, on = 'id')  # Merging the two datasets on the id column
# df2.head()

df2.info()
C = df2['vote_average'].mean()  # Finding the mean rating for all movies

C
# Minimum votes required to be listed in the chart

# We're using 90 percentile as our cutoff

# i.e. for the movie to be listed it must have more votes than at least 90% of the movies in the list

m = df2['vote_count'].quantile(0.9)  

m  
# Movies that qualify for our chart are filtered into q_movies

q_movies = df2.copy().loc[df2['vote_count'] >= m]

q_movies.shape
# We find there are 481 movies which qualify to be in this list.

# Calculating the metric for each qualified movie

# Defining funciton weighted_rating() and new feature score

def weighted_ratings(x, m=m, C=C):

    v = x['vote_count']  # No. of voted for the movie

    R = x['vote_average']  # Average rating for the movie

    # Calculation based on teh IMDB formula

    return ((v/v+m) * R + (m/m+v) * C)



# Weighed Rating (WR) = ((v/v+m) * R + (m/m+v) * C)

# v = number of votes for the movie;

# m = minimum votes required to be listed in the chart;

# R = average rating of the movie; And

# C = mean vote across the whole report

# Define a new feature 'score' and calculate its value with 'weighted_rating()'



q_movies['score'] = q_movies.apply(weighted_ratings, axis = 1)
# Sort movies based on score calculated above

q_movies = q_movies.sort_values('score', ascending = False)



# Print the top 15 movies

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
# Under the Trending Now tab, we find movies that are very popular 

# Sorting the dataset by popularity column

pop = df2.sort_values('popularity', ascending = False)

import matplotlib.pyplot as plt

plt.figure(figsize = (12, 4))



# Making a horizontal bar plot of popular movies

plt.barh(pop['title'].head(6), pop['popularity'].head(6), align = 'center', color = 'skyblue')

plt.gca().invert_yaxis()

plt.xlabel('Popularity')

plt.title('Popular Movies')