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
dfmovie=pd.read_csv('../input/movies.dat', sep = '::', engine='python')
dfmovie.columns =['MovieIDs','MovieName','Category']
dfmovie.head()
df_rating = pd.read_csv("../input/ratings.dat",sep='::', engine='python')
df_rating.columns=['UserID','MovieID','Ratings','TimeStamp']
#df_rating_matrix = pd.get_dummies(df_rating.set_index('ID')['P'].astype(str)).max(level=0).sort_index()
df_rating.head()
result = pd.concat([dfmovie, df_rating], axis=1)
#(result)
result.head()
#test =pd.get_dummies(df_rating.UserID).groupby(df_rating.MovieID).apply(max)
#test.head() 
#R = df_rating.values
new_rating_matrix = df_rating.pivot(index = 'UserID', columns ='MovieID', values = 'Ratings').fillna(0)
new_rating_matrix.head()
df_norm = (new_rating_matrix - new_rating_matrix.mean()) / (new_rating_matrix.max() - new_rating_matrix.min())
df_norm.head()
u, s, vh = np.linalg.svd(df_norm, full_matrices=True)
u.shape, s.shape, vh.shape
np.allclose(df_norm, np.dot(u[:, :3706] * s, vh))
from scipy.sparse.linalg import svds
n_u,n_s,n_vt = svds(df_norm, k = 40)
n_u.shape,n_s.shape,n_vt.shape


#from sklearn.metrics import mean_squared_error
#mean_squared_error(df_norm, df_norm)
new_a = np.dot(n_u, np.dot(np.diag(n_s), n_vt))
new_a

from sklearn.metrics import mean_squared_error
mean_squared_error(df_norm.values, new_a)

