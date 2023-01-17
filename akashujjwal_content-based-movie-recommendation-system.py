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
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline
movie=pd.read_csv('../input/Movie _data.csv')
movie.head()
movie['movieID'].value_counts()
movie.hist(column='movieID', bins=50)
movie.columns
movie.columns.drop('userID',1).drop('Name',1)
moviesWithGenres_df = movie.copy()

for index, row in movie.iterrows():
    for genre in row['Genre']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()
userInput = [
            {'Movie':'Sultan','rating':2},
           
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
#Filtering out the movies by title
inputId = movie[movie['Movie'].isin(inputMovies['Movie'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies
inputMovies['Rating']
userMovies = movie[movie['movieID'].isin(inputMovies['movieID'].tolist())]
userMovies
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieID'].isin(inputMovies['movieID'].tolist())]
userMovies
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('userID', 1).drop('Name', 1).drop('Movie', 1).drop('Rating', 1).drop('movieID',1).drop('ActorID',1).drop('Actor',1).drop('GenreID',1).drop('Genre',1)
userGenreTable
inputMovies['Rating']
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['Rating'])
#The user profile
userProfile
#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieID'])
#And drop the unnecessary information
genreTable = genreTable.drop('userID', 1).drop('Name', 1).drop('Movie', 1).drop('Rating', 1).drop('movieID',1).drop('ActorID',1).drop('Actor',1).drop('GenreID',1).drop('Genre',1)
genreTable.head()
genreTable.shape
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
recommendationTable_df.head()
#The final recommendation table
movie.loc[movie['movieID'].isin(recommendationTable_df.head(3).keys())].drop('userID',1).drop('Name',1).drop('Rating',1)