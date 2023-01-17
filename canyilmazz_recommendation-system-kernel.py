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
#import movie dataset at look at columns

movie=pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")

movie.columns
#we choose ıd and title

movie=movie.loc[:,["movieId","title"]]

movie.head()
#import rating dataset and look at columns

rating=pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")

rating.columns
#we choose userıd,movieıd and rating

rating=rating.loc[:,["userId","movieId","rating"]]

rating.head()
# then merge movie and rating data

data=pd.merge(movie,rating)

data.head()
data.shape
data=data.iloc[:1000000,:]
# lets make a pivot table in order to make rows are users and columns are movies

pivot_table=data.pivot_table(index=["userId"],columns=["title"],values="rating")

pivot_table.head(10)
x=pivot_table["Two Bits (1995)"]

similarity_movies=pivot_table.corrwith(x)

similarity_movies=similarity_movies.sort_values(ascending=False)

similarity_movies.head()