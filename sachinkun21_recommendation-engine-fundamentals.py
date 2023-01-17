# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
ratings  = pd.read_csv('../input/ratings_small.csv')

ratings.head()
print(ratings.shape)
from sklearn.model_selection import train_test_split

train_df,test_df = train_test_split(ratings, test_size = 0.3, random_state = 42)

print(train_df.shape, '\t\t', test_df.shape)
train_df.head()
df_movies_as_features = train_df.pivot(index = 'userId', columns = 'movieId',values = 'rating' )

df_movies_as_features.shape
df_movies_as_features.head()
df_movies_as_features.fillna(0, inplace = True)

df_movies_as_features.head()
from sklearn.metrics.pairwise import pairwise_distances