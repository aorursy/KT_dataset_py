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
import pandas as pd
pd.options.display.max_rows = 10
unames = ['user_id','gender','age','occupation','zip']

users = pd.read_table('../input/ml-1m/ml-1m/users.dat',sep='::',header = None, names = unames)
engine='python'
unames = ['user_id','gender','age','occupation','zip']

users = pd.read_table('../input/ml-1m/ml-1m/users.dat',sep='::',header = None, names = unames)
users[:5]
rnames = ['user_id','movie_id','rating','timestamp']

ratings = pd.read_table('../input/ml-1m/ml-1m/ratings.dat',sep='::',header = None, names = rnames)
rnames[:10]
ratings[:5]
mnames = ['movie_id','title','genres']

movies = pd.read_table('../input/ml-1m/ml-1m/movies.dat',sep='::',header = None, names = mnames)
movies[:10]
ratings
movies
data = pd.merge(pd.merge(ratings,users),movies)
data
data.iloc[0]
mean_ratings = data.pivot_table('rating',index='title',columns='gender',aggfunc='mean')
mean_ratings
ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]
active_titles = ratings_by_title.index[ratings_by_title >= 250]
active_titles
mean_ratings = mean_ratings.loc[active_titles]
mean_ratings
top_female_ratings= mean_ratings.sort_values(by='F',ascending =False)
top_female_ratings
mean_ratings['diff'] = mean_ratings['M']-mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by = 'diff')
sorted_by_diff
sorted_by_diff[::-1][:10]
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]
rating_std_by_title.sort_values(ascending = False)