# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

anime = pd.read_csv("../input/anime-recommendations-database/anime.csv")

rating = pd.read_csv("../input/anime-recommendations-database/rating.csv")
anime.head()

rating.head()
df = pd.merge(rating,anime,on='anime_id')

df.head()
df = df.drop('user_id', axis = True)

df.head()
combine_anime_rating = df.dropna(axis = 0, subset = ['name'])



anime_ratingCount = (combine_anime_rating.

     groupby(by = ['name'])['rating_y'].

     count().

     reset_index().

     rename(columns = {'rating_y': 'totalRatingCount'})

     [['name', 'totalRatingCount']]

    )

anime_ratingCount.head()
rating_with_totalRatingCount = combine_anime_rating.merge(anime_ratingCount, left_on = 'name', right_on = 'name', how = 'left')

rating_with_totalRatingCount.head()
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print(anime_ratingCount['totalRatingCount'].describe())
popularity_threshold = 50

rating_popular_anime = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

rating_popular_anime.tail()
rating_popular_anime.shape
## First lets create a Pivot matrix



anime_features_df = rating_popular_anime.pivot_table(index='name',columns='anime_id',values='rating_x').fillna(0)

anime_features_df.head()


from scipy.sparse import csr_matrix



anime_features_df_matrix = csr_matrix(anime_features_df.values)



from sklearn.neighbors import NearestNeighbors





model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')

model_knn.fit(anime_features_df_matrix)
anime_features_df.shape
import numpy as np

query_index = np.random.choice(anime_features_df.shape[0])

print(query_index)

distances, indices = model_knn.kneighbors(anime_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 9)
for i in range(0, len(distances.flatten())):

    if i == 0:

        print('Recommendations for {0}:\n'.format(anime_features_df.index[query_index]))

    else:

        print('{0}: {1}, with distance of {2}:'.format(i, anime_features_df.index[indices.flatten()[i]], distances.flatten()[i]))