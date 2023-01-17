import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy.sparse import csr_matrix as sparse_matrix #create sparse matrix

from sklearn.neighbors import NearestNeighbors# our collaborative filtering?



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
#reading the book data and create columns of users, books and their ratings

filename = "../input/updated-good-reads-book/ratings.csv"

with open(os.path.join(filename), "rb") as f:

    ratings = pd.read_csv(f)



with open(os.path.join("../input/updated-good-reads-book/books.csv"), "rb") as f:

    books = pd.read_csv(f)



#getting the set of unique users and books to set the boundaries of the sparse matrix

n = len(set(ratings["user_id"]))

d = len(set(ratings["book_id"]))

print("Is book id set from ratings ", d, " the same as the books in the books table ?", books.shape)



user_mapper = dict(zip(np.unique(ratings["user_id"]), list(range(n))))

item_mapper = dict(zip(np.unique(ratings["book_id"]), list(range(d))))

# user_inverse_mapper = dict(zip(list(range(n)), np.unique(ratings["user_id"])))

# item_inverse_mapper = dict(zip(list(range(d)), np.unique(ratings["book_id"])))

user_ind = [user_mapper[i] for i in ratings["user_id"]]

item_ind = [item_mapper[i] for i in ratings["book_id"]]

Xsparse = sparse_matrix((ratings["rating"], (user_ind, item_ind)), shape=(n,d))

neigh = NearestNeighbors()

neigh.fit(Xsparse.transpose())

#our sample book is Harry potter, so recommended item should also be Harry Potter related

book = "Harry Potter and the Sorcerer's Stone (Harry Potter, #1)"

#I have an off by 1 error here, the books_id starts at 1, but in the ratings sparse matrix it's 0 based, so take off 1 here

book_id = books.loc[books['title'] == book]["book_id"] -1

#Take the column of harry potter book id as our input

test_vec = Xsparse[:,book_id]

#transform the column into a vector and look for 10 suggested items

y_pred=neigh.kneighbors(test_vec.transpose(),10)

#prediction includes the distance of the recommended book from the given book as well as the distances

print(y_pred)



#Here we only care about getting the book ids and printing out the suggested books

predictItemDefault = y_pred[1][0][1:]

print(predictItemDefault)

for i in range(9):

    #Same 0 based indexing issue from before applies

    print(books.loc[books['book_id'] == predictItemDefault[i]+1]["title"])
