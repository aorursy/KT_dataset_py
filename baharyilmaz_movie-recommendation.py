



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import movie data

data_movie=pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")

data_movie.columns
data_movie=data_movie.loc[:,["movieId","title"]]

data_movie.head()
# import rating data

data_rating=pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")

data_rating.columns
data_rating = data_rating.loc[:,["userId","movieId","rating"]]

data_rating.head()
# merge movie and rating data

data=pd.merge(data_movie,data_rating)

data.head()
data.shape

# use 1 million of sample in data

data=data.iloc[:1000000,:]
# pivot table

pivot_table=data.pivot_table(index=["userId"],columns = ["title"],values = "rating")

pivot_table.head()
# find similarities between a movie and other movies

movie_watched =pivot_table["Bad Boys (1995)"]

similarity=pivot_table.corrwith(movie_watched) #find correlation 

similarity=similarity.sort_values(ascending=False)

similarity.head()