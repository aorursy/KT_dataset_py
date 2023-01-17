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
import math

import matplotlib.pyplot as plt
ratings = pd.read_csv("/kaggle/input/goodbooks-10k/ratings.csv")
ratings.head(2)
n_users = ratings.user_id.unique().shape[0]

n_books = ratings.book_id.unique().shape[0]

print("The database contains ratings from", n_users, "users for", n_books, "books.")
data_matrix = np.zeros((n_users, n_books))

for line in ratings.itertuples():

    data_matrix[line[2]-1, line[1]-1] = line[3]
user_row = 23 # set user_id to explore here
n_books_rated = 0

for book in range(0, n_books):

    if data_matrix[user_row, book] != 0:

        n_books_rated += 1

print(n_books_rated)
np.count_nonzero(data_matrix[user_row]) # alternative method using numpy library
books_rated = np.zeros(n_users)

for user in range(0, n_users):

    books_rated[user] = np.count_nonzero(data_matrix[user])
plt.figure(figsize=(15,5))

plt.hist(books_rated, bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 200])
def cos_similar(u, v):

    return np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v)
cos_similar(data_matrix[2000], data_matrix[2000]) # similarity of itself
cos_similar(data_matrix[2000], data_matrix[2001]) # similarity of itself
similar = np.zeros(n_users)

for user in range(0, n_users):

    similar[user] = cos_similar(data_matrix[user_row], data_matrix[user])
print("This user read", np.count_nonzero(data_matrix[user_row]), "books.")

print("There are", np.count_nonzero(similar), "users that read at least one same book as this user.")
plt.figure(figsize=(15,5))

plt.hist(similar, bins = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
top_similar = np.where(similar > 0.2)[0] # top similar users

print(top_similar)
for user in top_similar:

    if user != user_row:

        print("Other similar users/recommenders read", np.count_nonzero(data_matrix[user]), "books.")
books_rating = np.zeros(n_books)

for user in top_similar:

    books_rating += data_matrix[user]/len(top_similar)

plt.hist(books_rating, bins = [1,2,3,4,5,6])
best_book = np.where(books_rating == max(books_rating))[0][0]

print(best_book)
good_book = np.where(books_rating > max(books_rating)/4)[0]

print(good_book, len(good_book))
read_book = np.where(data_matrix[user_row] != 0)[0]

print(read_book)
books = pd.read_csv("/kaggle/input/goodbooks-10k/books.csv")

books.head(2)
print("The user rated the following books:")

books.loc[books['id'].isin(read_book)]
recommend_book = np.delete(good_book, np.where(np.isin(good_book, read_book)))
print("Recommended books for the users are:")

books.loc[books['id'].isin(recommend_book)]