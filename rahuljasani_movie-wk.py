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
movies_df = pd.read_csv('/kaggle/input/movie-recommendation-system/movies.csv')

ratings_df = pd.read_csv('/kaggle/input/movie-recommendation-system/ratings.csv')

movies_df
ratings_df
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)

movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)

movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df.head()
movies_df['genres'] = movies_df.genres.str.split('|')

movies_df.head()
moviesWithGenres_df = movies_df.copy()

for index, row in movies_df.iterrows():

    for genre in row['genres']:

        moviesWithGenres_df.at[index, genre] = 1

moviesWithGenres_df = moviesWithGenres_df.fillna(0)

moviesWithGenres_df.head()
ratings_df.head()
ratings_df = ratings_df.drop('timestamp', 1)

ratings_df.head()
userInput = [

            {'title':'Breakfast Club, The', 'rating':5},

            {'title':'Toy Story', 'rating':3.5},

            {'title':'Jumanji', 'rating':2},

            {'title':"Pulp Fiction", 'rating':5},

            {'title':'Akira', 'rating':4.5}

         ] 

inputMovies = pd.DataFrame(userInput)

inputMovies
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

inputMovies = pd.merge(inputId, inputMovies)

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

inputMovies
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies
userMovies = userMovies.reset_index(drop=True)

userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userGenreTable
inputMovies['rating']
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

userProfile
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

genreTable.head()
genreTable.shape
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendationTable_df.head()
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

recommendationTable_df.head()
movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]