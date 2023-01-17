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
# import movie dataset and let's look at the columns

movie = pd.read_csv("../input/movie.csv")

movie.columns
movie.info()
movie = movie.loc[:,['movieId', 'title']]

movie.head(10)
# import rating dataset and look at the columns

rating = pd.read_csv("../input/rating.csv")

rating.columns


rating = rating.loc[:,["userId", "movieId", "rating"]]

rating.head(10)
# merge movie and rating data

veri = pd.merge(movie,rating)
# Now let's check out the data

veri.head(10)
veri.shape
veri = veri.iloc[:500000,:]
pivotTablo = veri.pivot_table(index = ["userId"], columns = ["title"],values = "rating")

pivotTablo.head(10)
movie_watched = pivotTablo["Toy Story (1995)"]

similarity_with_other_movies = pivotTablo.corrwith(movie_watched)  # find correlation between "Bad Boys (1995)" and other movies

similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending=False)

similarity_with_other_movies.head()