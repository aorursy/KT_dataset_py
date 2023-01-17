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
movie = pd.read_csv("../input/movie.csv")
movie.columns
movie = movie.loc[:,["movieId","title"]]
movie.head(10)
rating = pd.read_csv("../input/rating.csv")
rating.columns
rating = rating.loc[:,["userId","movieId","rating"]]
rating.head()
data = pd.merge(movie,rating)
data.head(10)
data.shape
data = data.iloc[:1000000,:]
pivot_table =data.pivot_table(index = ["userId"],columns = ["title"],values = ["rating"])
pivot_table.head(10)
# moviw_watched = pivot_table["Bad Boys"]
# similarity_with_other_movies = pivot_table.corrwith(movie_watched)
# similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)
# similarity_with_other_movies.head()
