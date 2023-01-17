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
movies=pd.read_csv('/kaggle/input/movie-recommendation-system/movies.csv')

ratings=pd.read_csv('/kaggle/input/movie-recommendation-system/ratings.csv')
movies
ratings
movies.head()
movies['genres']=movies.genres.str.split('|')

movies.head()
moviesWithGenres=movies.copy()

for index,row in movies.iterrows():

    for genre in row['genres']:

        moviesWithGenres.at[index,genre]=1

moviesWithGenres=moviesWithGenres.fillna(0)

moviesWithGenres.head()
ratings.head()
ratings=ratings.drop('timestamp',1)

ratings.head()
userInput=[{'title':'Toy Story','rating':4.5},{'title':'Jumanji','rating':4.7},{'title':'Grumpier Old Men','rating':3.8},{'title':'Waiting to Exhale','rating':4.8},{'title':'Father of the Bride Part II','rating':3.0}]

inputMovies=pd.DataFrame(userInput)

inputMovies
inputId=movies[movies['title'].isin(inputMovies['title'].tolist())]

inputMovies=pd.merge(inputId,inputMovies)

inputMovies=inputMovies.drop('genres',1)

inputMovies
userMovies=moviesWithGenres[moviesWithGenres['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies
userMovies=userMovies.reset_index(drop=True)

userGenreTable=userMovies.drop('movieId',1).drop('title',1)

userGenreTable
inputMovies['rating']
genreTable=moviesWithGenres.set_index(moviesWithGenres['movieId'])

genreTable=genreTable.drop('movieId',1).drop('title',1).drop('genres',1)

genreTable.head()
genreTable.shape
userProfile=userGenreTable.transpose().dot(inputMovies['rating'])

userProfile
recommendationTable=((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
movies.loc[movies['movieId'].isin(recommendationTable.head().keys())]