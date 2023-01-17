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
scores=pd.read_csv('/kaggle/input/movielens-20m-dataset/genome_scores.csv')
movie=pd.read_csv('/kaggle/input/movielens-20m-dataset/movie.csv')
scores.head()
scores.shape
scores['movieId'].unique()
scores['movieId'].nunique()
scores['tagId'].unique()
scores['tagId'].nunique()
scores.describe()
movie_tag_pivot=pd.pivot_table(columns='tagId',index='movieId',values='relevance',data=scores)
movie_tag_pivot
movie_tag_pivot.fillna(-100,inplace=True)
from sklearn.neighbors import  NearestNeighbors
nn=NearestNeighbors(algorithm='brute',metric='minkowski',p=8)
nn.fit(movie_tag_pivot)
def recommend(movie_id):
    distances,suggestions=nn.kneighbors(movie_tag_pivot.loc[movie_id,:].values.reshape(1,-1),n_neighbors=16)
    return movie_tag_pivot.iloc[suggestions[0]].index.to_list()
scores_movie=movie[movie['movieId'].isin(movie_tag_pivot.index.to_list())]
scores_movie[scores_movie['title'].str.contains('avengers',case=False)]
recommendations=recommend(89745)
recommendations
for movie_id  in recommendations[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])
scores_movie[scores_movie['title'].str.contains('harry potter',case=False)]
recommendations=recommend(4896)
for movie_id  in recommendations[1:]:
    print(movie[movie['movieId']==movie_id]['title'].values[0])
import pickle
pickle.dump(nn,open('engine.pkl','wb'))
pickle.dump(movie_tag_pivot,open('movie_tag_pivot_table.pkl','wb'))
pickle.dump(scores_movie,open('movie_names.pkl','wb'))
