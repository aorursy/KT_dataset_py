

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import sklearn
from sklearn.decomposition import TruncatedSVD
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/ratings.csv');df.head()
df.shape
len(df.movieId.unique())
df.describe()
sub_df = df[df['userId'] < 300]; len(sub_df.movieId.unique())
sub_df.groupby('movieId')['rating'].count().sort_values(ascending = False).head()
filter = (sub_df['movieId']==296)
cross_rate = sub_df.pivot_table(values = 'rating', index = 'userId', columns = 'movieId', fill_value = 0);cross_rate.head()
cross_rate.shape
X= cross_rate.T; X.shape
SVD = TruncatedSVD(n_components = 12, random_state = 17)
matrix = SVD.fit_transform(X); matrix.shape
corr = np.corrcoef(matrix);corr[:5]
movie_296 = corr[296];movie_296.shape
np.where((movie_296>0.9) & (movie_296 <1.0))
sub_df.groupby('userId')['movieId'].count().sort_values(ascending = False).head()
user_156 = sub_df[(sub_df['userId'] == 156)]['movieId']
user_matrix = SVD.fit_transform(cross_rate); user_matrix.shape
corr = np.corrcoef(user_matrix)
user_156 = corr[156]
np.where((user_156>0.9) & (user_156 <1.0))
fellow = (np.where((user_156>0.9) & (user_156 <1.0)))[0].tolist();fellow
sub_df[(sub_df['userId'] == 1) & (sub_df['rating']>4)]['movieId'].count()
movie_id = cross_rate.columns
recommend = list()
for i in fellow:
    fellow_love = sub_df[(sub_df['userId'] == i) & (sub_df['rating']>4)]['movieId']
    for love in fellow_love:
        if love not in user_156: 
            recommend.append(love)

recommend