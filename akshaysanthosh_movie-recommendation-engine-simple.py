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
import numpy as np
import pandas as pd
df1=pd.read_csv('../input/tmdb_5000_credits.csv')
df2=pd.read_csv('../input/tmdb_5000_movies.csv')
df1.head(5)
df2.head(5)
df1.columns = ['id','tittle','cast','crew']


df1.head(5)
df2= df2.merge(df1,on='id')
df2.head(5)
C= df2['vote_average'].mean()
C
m= df2['vote_count'].quantile(0.9)
m

q_movies = df2.copy().loc[df2['vote_count'] >= 1838]
q_movies.shape

q_movies.head(3)
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
