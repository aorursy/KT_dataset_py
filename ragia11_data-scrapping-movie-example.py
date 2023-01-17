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
import pandas as pd
## Read the user data
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv( 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user',  sep='|', names=u_cols)

users.head()
#Read the ratings
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t',names=r_cols )

ratings.head()
# data about the movies
# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date',   'video_release_date', 'imdb_url']

movies = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item',  sep='|', names=m_cols, encoding = 'ISO-8859-1', usecols=range(5))

movies.head()
 # Get information about data
print (movies.dtypes)
print()
print(movies.describe()) 
users.head()
users['occupation'].tail()
users.iloc[3]
print( users[  ['occupation', 'sex'] ].tail() )

columns_you_want = ['occupation', 'sex','age'] 

print( users[ columns_you_want ].tail() )

print( users[users.age >25] )
print(users['age'] >25 )
ou= users[users.age >25]
print(ou.head() )
print( users[ (users.age == 40) & (users.sex=='M') ] )
print( users[ (users.occupation  == 'programmer') & (users.sex=='F') ]  ) 
print()
selected_users =users[ (users.occupation  == 'programmer') & (users.sex=='F') ]
print ( selected_users.describe() )
print()
print (selected_users.mean() )
print(ratings.head())
grouped_data = ratings.groupby('user_id')
#grouped_data =ratings['movie_id'].groupby(ratings['user_id'])
ratng_per_user= grouped_data.count()
print( 'ratng_per_user'   )
print(  ratng_per_user  )
grouped_data_by_movie =ratings['rating'].groupby(ratings['movie_id'])
movies_ratngs_mean= grouped_data_by_movie.mean()
print('avg ratngs')
print(movies_ratngs_mean.head() )
#advanced: get the movie titles with the highest average rating

print('the movie titles with the highest average rating')
maximum_rating =movies_ratngs_mean.max()

print(maximum_rating)
print(movies_ratngs_mean.head() ) 
good_movie_ids = movies_ratngs_mean[movies_ratngs_mean == maximum_rating].index
print('good_movie_ids')

print(good_movie_ids)

print (  "Best movie titles" )
print ( movies[movies.movie_id.isin(good_movie_ids)].title )

movies.head()
how_many_ratings_per_movie= grouped_data_by_movie.count()
print('how_many_ratings_per_movie')
print(how_many_ratings_per_movie)
print('how_many_ratings_per max  or good movie')

print(how_many_ratings_per_movie[movies_ratngs_mean == maximum_rating] )

average_ragings = grouped_data_by_movie.apply(lambda f: f.mean())
average_ragings.head()
grouped_data = ratings.groupby('user_id')
avg_per_user= grouped_data.mean()
print( 'avg_per_user'   )
print(  avg_per_user.head()  )
ocup_user=users['sex'].groupby(users['occupation'])
male_dominant_occupations = ocup_user.apply(lambda f: sum(f == 'M') > sum(f == 'F'))
print( male_dominant_occupations)
print ('\n')
ocup_user=users[[ 'sex','occupation']].groupby(users['occupation']) 
print (ocup_user.head())
 # get numbers of each 
