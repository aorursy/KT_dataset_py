import numpy as np
import pandas as pd
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff

from PIL import Image
import requests
from io import BytesIO
#Users
u_cols = ['user_id', 'location', 'age']
users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)

#Books
i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)

#Ratings
r_cols = ['user_id', 'isbn', 'rating']
ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)
users = users.iloc[1:]
books = books.iloc[1:]
ratings = ratings.iloc[1:]
ratings['rating'] = ratings['rating'].astype(int)
dat = ff.create_table(users.head())
dat.update_layout(autosize=False,height=200, width = 700)
dat = ff.create_table(ratings.head())
dat.update_layout(autosize=False,height=200, width = 700)
dat = ff.create_table(books.head())
dat.update_layout(autosize=False,height=200, width = 4400)
df = pd.merge(ratings,books,on='isbn')
dat = ff.create_table(df.head())
dat.update_layout(autosize=False,height=200, width = 3990)
df['rating'] = df['rating'].astype(int) 
ratings = pd.DataFrame(df.groupby('book_title')['rating'].mean())
ratings['Total_Ratings'] = pd.DataFrame(df.groupby('book_title')['rating'].count())
Top5books = ratings[ratings['Total_Ratings'] > 30].sort_values(by = 'rating', ascending = False)[:5].reset_index()
Top5books1 = pd.merge(Top5books, df, on = 'book_title', how = 'inner')
dat = ff.create_table(Top5books)
dat.update_layout(autosize=False,height=200, width = 1200)
pandas_profiling.ProfileReport(df)
dat = ff.create_table(df.head())
dat.update_layout(autosize=False,height=200, width = 3990)
df.drop(['year_of_publication', 'publisher', 'book_author', 'img_s', 'img_m', 'img_l'], axis = 1, inplace = True)
dat = ff.create_table(df.head())
dat.update_layout(autosize=False,height=200, width = 800)
combine_book_rating = df.dropna(axis = 0, subset = ['book_title'])

book_ratingCount = (combine_book_rating.
     groupby(by = ['book_title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['book_title', 'totalRatingCount']]
    )

dat = ff.create_table(book_ratingCount.head())
dat.update_layout(autosize=False,height=200, width = 1200)
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'book_title', right_on = 'book_title', how = 'left')

dat = ff.create_table(rating_with_totalRatingCount.head())
dat.update_layout(autosize=False,height=200, width = 1200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(book_ratingCount['totalRatingCount'].describe())
print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))
#threshold
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
dat = ff.create_table(rating_popular_book.head())
dat.update_layout(autosize=False,height=200, width = 900)
#to avoid memory issue:
users['location'].value_counts()
combined = rating_popular_book.merge(users, left_on = 'user_id', right_on = 'user_id', how = 'left')

all_user_rating = combined[combined['location'].str.contains("usa|canada|united kingdom|australia")]
all_user_rating=all_user_rating.drop('age', axis=1)

dat = ff.create_table(all_user_rating.head())
dat.update_layout(autosize=False,height=200, width = 1500)
all_user_rating = all_user_rating.drop_duplicates(['user_id', 'book_title'])
all_user_rating_pivot = all_user_rating.pivot(index = 'book_title', columns = 'user_id', values = 'rating').fillna(0)
print('No null values now..')
all_user_rating_matrix = csr_matrix(all_user_rating_pivot.values)
print('Sparse matrix created..')
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(all_user_rating_matrix)
query_index = np.random.choice(all_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(all_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(all_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, all_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
query_index = np.random.choice(all_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(all_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(all_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, all_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
query_index = np.random.choice(all_user_rating_pivot.shape[0])
distances, indices = model_knn.kneighbors(all_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(all_user_rating_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, all_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))