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
rating = pd.read_csv("../input/rating.csv")
movie = movie.loc[:,["movieId","title"]]
rating = rating.loc[:,["userId","movieId","rating"]]
data = pd.merge(movie,rating)
data = data.iloc[:1000000,:]
data.tail()
pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")
pivot_table.head(10)
movie_watched = pivot_table["Bad Boys (1995)"]
similarity_with_other_films = pivot_table.corrwith(movie_watched)
similarity_with_other_films = similarity_with_other_films.sort_values(ascending=False)
similarity_with_other_films.head()












































