# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
userInfo = ['userId','movieId','rating','timestamp']
users = pd.read_csv('../input/ml-100k/u.data', sep='\t', names=userInfo)
users.head(10)
movieInfo = ['movieId', 'title']
movies = pd.read_csv('../input/ml-100k/u.item', sep='|', encoding="437", 
                        index_col=False, names=movieInfo, usecols=[0,1])
movies.head(10)
ratings = pd.merge(users, movies,left_on='movieId',right_on="movieId")
ratings.head(10)
#useful pandas commands
ratings.loc[0:10,['userId']]
ratings = pd.DataFrame.sort_values(ratings,['userId','movieId'],ascending=[0,1])
ratings.head(10)

# Check how many movies were rated by each user, and the number of users that rated each movie 
moviesPerUser = ratings.userId.value_counts()
print (moviesPerUser)
usersPerMovie = ratings.title.value_counts()
print (usersPerMovie)