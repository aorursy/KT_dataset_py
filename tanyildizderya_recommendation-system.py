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
movie = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
movie.columns
movie = movie.loc[:,["movieId","title"]]

movie.head(10)
ratings = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")

ratings.columns
ratings = ratings.loc[:,["userId","movieId","rating"]]

ratings.head()
data = pd.merge(movie,ratings)

data.head(10)
data.shape
data = data.iloc[:1000000,:]
pivot_table = data.pivot_table(index=["userId"],columns=["title"],values="rating")

pivot_table.head(10)
movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_movies = pivot_table.corrwith(movie_watched) 
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()
movie_watched = pivot_table["Vampire in Brooklyn (1995)"]

similarity_with_other_movies = pivot_table.corrwith(movie_watched)

similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)

similarity_with_other_movies.head()
movie_watched = pivot_table["Up Close and Personal (1996)"]

similarity_with_other_movies = pivot_table.corrwith(movie_watched)

similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)

similarity_with_other_movies.head()