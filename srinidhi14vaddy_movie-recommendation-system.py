# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from math import sqrt
import matplotlib.pyplot as plt
%matplotlib inline
movies = pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')
ratings = pd.read_csv('/kaggle/input/movielens-20m-dataset/rating.csv')
movies.shape
ratings.shape
movies.head()
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies['year'] = movies.year.str.extract('(\d\d\d\d)',expand=False)

movies.head()
#Removing the years from the 'title' column
movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies['title'] = movies['title'].apply(lambda x: x.strip())
movies.head()
movies['genres'] = movies.genres.str.split('|')
movies.head()
moviesGenres = movies.copy()
moviesGenres.head()
print(movies.iterrows())
#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies.iterrows():
    for genre in row['genres']:
        moviesGenres.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesGenres = moviesGenres.fillna(0)
moviesGenres.head()
ratings.head()
ratings=ratings.drop('timestamp',1)
ratings.head()
userInput = [
            {'title':'Breakfast Club, The', 'rating':4},
            {'title':'Toy Story', 'rating':4.5},
            {'title':'Jumanji', 'rating':2.5},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':3.5}
         ] 
userMovie = pd.DataFrame(userInput)
userMovie
#Filtering out the movies by title
inputId = movies[movies['title'].isin(userMovie['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
userMovie = pd.merge(inputId, userMovie)
userMovie
userMovie=userMovie.drop('genres',1)
userMovie=userMovie.drop('year',1)
userMovie
genre = moviesGenres[moviesGenres['movieId'].isin(userMovie['movieId'].tolist())]
genre
genretable=genre.copy()
genretable=genretable.reset_index(drop=True)
genretable=genretable.drop('title',1).drop('movieId',1).drop('genres',1).drop('year',1)
genretable
userMovie['rating']
#Dot produt to get weights
user = genretable.transpose().dot(userMovie['rating'])
#The user profile
user
recommendationtable = moviesGenres.set_index(moviesGenres['movieId'])
recommendationtable.head()
recommendationtable = recommendationtable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)
recommendationtable.head()
recommend = (recommendationtable*user).sum(axis=1)/(user.sum())
recommend.head()
recommend=recommend.sort_values(ascending=False)
recommend.head()
movies.loc[movies['movieId'].isin(recommend.head(20).keys())]
