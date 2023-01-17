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
#exploring movie data set

moviedata = pd.read_csv('../input/movie.csv')

moviedata.columns
moviedata = moviedata.loc[:,['movieId','title']]

moviedata.head(10)

#import rating data 

movierating = pd.read_csv('../input/rating.csv')

movierating.columns
movierating = movierating.loc[:,['userId','movieId','rating']]

movierating.head()
#mering movie and movie rating



data = pd.merge(moviedata,movierating)

data.head()
data.shape
data = data.iloc[:1000000,:]
# now create a table, user in rows and movies in colums



pivot_table = data.pivot_table(index = ["userId"],columns = ["title"],values = "rating")

pivot_table.head(10)
movie_watched = pivot_table['Usual Suspects, The (1995)']

# find correlation between "Usual Suspects, The (1995)" and other movies

similar_other_movies = pivot_table.corrwith(movie_watched)

similar_other_movies = similar_other_movies.sort_values(ascending = False)

similar_other_movies.head()