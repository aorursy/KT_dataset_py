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
import numpy as np
rating_df = pd.read_csv( "../input/u.data", delimiter = "\t", header = None )
rating_df.head( 10 )
rating_df.columns = ["userid", "movieid", "rating", "timestamp"]
rating_df.head( 10 )
len( rating_df.userid.unique() )
len( rating_df.movieid.unique() )
rating_df.drop( "timestamp", inplace = True, axis = 1 )
rating_df.head( 10 )
movies_df = pd.read_csv( "../input/u.item", delimiter = '\|', header = None )
movies_df = movies_df.iloc[:,:2]
movies_df.columns = ['movieid', 'title']
movies_df.head( 10 )
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation
user_movies_df = rating_df.pivot( index='userid', columns='movieid', values = "rating" ).reset_index(drop=True)
user_movies_df.fillna(0, inplace = True)
user_movies_df.shape
user_movies_df.iloc[10:20, 20:30]
from IPython.display import Image
Image("../input/recomend-1.png")
from IPython.display import Image
Image("../input/recomend-2.png")
user_sim = 1 - pairwise_distances( user_movies_df.as_matrix(), metric="cosine" )
user_sim_df = pd.DataFrame( user_sim )
user_sim_df[0:5]
user_sim_df.idxmax(axis=1)[0:5]
np.fill_diagonal( user_sim, 0 )
user_sim_df = pd.DataFrame( user_sim )
user_sim_df[0:5]
user_sim_df.idxmax(axis=1).sample( 10, random_state = 10 )
def get_user_similar_movies( user1, user2 ):
  common_movies = rating_df[rating_df.userid == user1].merge(rating_df[rating_df.userid == user2], on = "movieid", how = "inner" )

  return common_movies.merge( movies_df, on = 'movieid' )
get_user_similar_movies( 310, 247 )
rating_mat = rating_df.pivot( index='movieid', columns='userid', values = "rating" ).reset_index(drop=True)
rating_mat.fillna( 0, inplace = True )
rating_mat.shape
rating_mat.head( 10 )
movie_sim = 1 - pairwise_distances( rating_mat.as_matrix(), metric="correlation" )
movie_sim.shape
movie_sim_df = pd.DataFrame( movie_sim )
movie_sim_df.head( 10 )
movies_df['similarity'] = movie_sim_df.iloc[0]
movies_df.columns = ['movieid', 'title', 'similarity']
movies_df.head( 10 )
movies_df.sort_values(by='similarity', ascending=False)[1:10]
def get_similar_movies( movieid, topN = 5 ):
  movies_df['similarity'] = movie_sim_df.iloc[movieid -1]
  top_n = movies_df.sort_values( ["similarity"], ascending = False )[0:topN]
  print( "Similar Movies to: ", )
  return top_n
get_similar_movies( 118 )
get_similar_movies( 127, 10 )
