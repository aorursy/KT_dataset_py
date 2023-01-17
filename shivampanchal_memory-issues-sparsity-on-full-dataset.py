# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
books = pd.read_csv("../input/Books.csv", encoding="ISO 8859-1")

users = pd.read_csv("../input/Users.csv", encoding="ISO 8859-1")

book_ratings = pd.read_csv("../input/BookRatings.csv", encoding="ISO 8859-1", low_memory=False)
book_ratings['UserID'] = book_ratings.UserID.astype(int) # Convert to int for User ID

grouped_cleaned = book_ratings.groupby(['UserID', 'ISBN']).sum().reset_index() # Group together

grouped_cleaned = grouped_cleaned.query('BookRating > 0') # Only get users where ratings totals were positive
grouped_cleaned.shape
import scipy.sparse as sparse

from scipy.sparse.linalg import spsolve

users = list(np.sort(grouped_cleaned.UserID.unique())) # Get our unique customers

books = list(grouped_cleaned.ISBN.unique()) # Get our unique products that were purchased

ratings = list(grouped_cleaned.BookRating) # All of our purchases
rows = grouped_cleaned.UserID.astype('category', categories = users).cat.codes 

# Get the associated row indices

cols = grouped_cleaned.ISBN.astype('category', categories = books).cat.codes 

# Get the associated column indices

ratings_sparse = sparse.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(books)))
ratings_sparse
matrix_size = ratings_sparse.shape[0]*ratings_sparse.shape[1] # Number of possible interactions in the matrix

num_ratings = len(ratings_sparse.nonzero()[0]) # Number of items interacted with

sparsity = 100*(1 - (num_ratings/matrix_size))

sparsity