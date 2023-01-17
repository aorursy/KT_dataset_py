import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

ratings = pd.read_csv('../input/ratings.csv')  #goes south with 10k books 1M recoomendations

books = pd.read_csv('../input/books.csv')

print( ratings.describe() )

print( books.head() )
from scipy.sparse import csr_matrix



user_u = list(ratings.user_id.unique())

book_u = list(ratings.book_id.unique())  #sorted automatically



col = ratings.user_id.astype('category', categories=user_u).cat.codes

row = ratings.book_id.astype('category', categories=book_u).cat.codes



bookrating = csr_matrix((ratings['rating'].tolist(), (row,col)), shape=(len(book_u),len(user_u)))

bookrating

from scipy.spatial.distance import cosine

from sklearn.metrics.pairwise import cosine_similarity

       

similarities = cosine_similarity(bookrating)  #goes south with 10k books

print(similarities.shape)

similarities
similaritiespd = pd.DataFrame(similarities,index=books.original_title)  #add titles



similar_books=pd.DataFrame(books.original_title)

for xi in range(0,10):

    similar_books[xi]=''



#example Twilight

tmp=similaritiespd.loc[:,2:2] #.sort_values(ascending=False)[:10])

print(tmp.sort_values(2,ascending=False)[:10])



for i in range(0,100):

    tmp= similaritiespd.sort_values(i,ascending=False)[:10].index

    for xi in range(0,10):

        similar_books.iat[i,xi] = tmp[xi]

    

similar_books