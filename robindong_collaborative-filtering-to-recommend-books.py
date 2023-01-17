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

rating = pd.read_csv('../input/ratings.csv')
# Show head of ratings.csv
rating.head()
# Lets look at the number of book_id and user_id

nr_books = len(rating['book_id'].unique())
nr_users = len(rating['user_id'].unique())
print(nr_books, nr_users)
# Use Coordinate Matrix of scipy to create sparse matrix for books and users
from scipy.sparse import coo_matrix

matrix = coo_matrix((rating['rating'].astype(float),
                     (rating['book_id'], rating['user_id'])))
print(matrix)
# Now we could do Singular-Value-Decomposition on the sparse matrix above

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2000, n_iter=10, random_state=23)
sigma = svd.fit_transform(matrix)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(sigma.shape)

# As you can see, we get a matrix with shape(10001, 2000), every row of this matrix represents a feature vector of a book
from sklearn.metrics.pairwise import cosine_similarity

# Create similarities for every book
udf = pd.DataFrame(cosine_similarity(sigma))
udf.head()
# With similarities of every book, we could search similar books for book_id==968, which is Dan Brown's 'The Da Vinci Code'

associate_books = udf.iloc[968].sort_values(ascending = False)

books = pd.read_csv('../input/books.csv')

list = []
for idx in associate_books.index[:100]:
    res = books.loc[books['book_id'] == idx]
    if not res.empty:
        print(res[['title', 'authors']])