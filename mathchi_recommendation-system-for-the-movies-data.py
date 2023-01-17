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
credits = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

movies = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")

ratings = pd.read_csv("../input/the-movies-dataset/ratings_small.csv")
# column name changed



credits.columns = ['movieId','title','cast','crew']



movies.rename(columns={"id": "movieId"}, inplace=True)

#movies.columns


ratings = ratings.loc[:,["userId","movieId","rating"]]

ratings.head(10)
data = pd.merge(credits,movies, on="movieId")

df = data.merge(ratings, on="movieId")

df.columns
df.drop(['status', 'title_x', 'title_y'], axis=1, inplace=True)             # benzer attribute lari sildik
df.head(1)
df.vote_average.mean()
# what we need is that movie id and title

df = df.loc[:,["movieId","original_title","userId","vote_average"]]

df.head(10)
df.shape


pivot_table = df.pivot_table(index = ["userId"],columns = ["original_title"],values = "vote_average")

pivot_table.head(100)
movie_watched = pivot_table["2001: A Space Odyssey"]





similarity_with_other_movies = pivot_table.corrwith(movie_watched)           

similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)

similarity_with_other_movies.head()