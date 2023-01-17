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
## importing datasets

users=pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Users.csv',sep=";",error_bad_lines=False, encoding='latin-1')

books = pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Books.csv',sep=";",error_bad_lines=False, encoding='latin-1')

rating=pd.read_csv('/kaggle/input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv',sep=";",error_bad_lines=False, encoding='latin-1')
books.head()
books.columns
books=books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']] #feature engineering : selecting features
books.head(2)
books.rename(columns={'Book-Title':'title','Book-Author':'author','Year-Of-Publication':'year','Publisher':'publisher'},inplace=True) #feature engineering : changing the column names
books.head(2)
users.head(2)
users.rename(columns={'User-ID':'user_id','Location':'location','Age':'age'},inplace=True) #feature engineering : changing the column names
rating.head(2)
rating.rename(columns={'User-ID':'user_id','Book-Rating':'rating'},inplace=True) #feature engineering : changing the column names
rating.head(2)
books.shape
users.shape
rating.shape
## unique users

rating['user_id'].value_counts().shape
## considering those users who have viewed more than 200 books

x=rating['user_id'].value_counts()>200

x[x].shape
y=x[x].index
rating=rating[rating['user_id'].isin(y)]
rating.shape
rating_with_books=rating.merge(books,on='ISBN') ##those users who have viewed more than 200 books with there rating on books
rating_with_books.shape
number_rating=rating_with_books.groupby('title')['rating'].count().reset_index() ## total rating of a book 
number_rating.rename(columns={'rating':'number of rating'},inplace=True) #feature engineering : changing the column names
number_rating.head()
final_ratings=rating_with_books.merge(number_rating,on='title') 
final_ratings.head()
final_ratings.shape
final_ratings=final_ratings[final_ratings['number of rating']>=50] ## considering those books which has got more than 50 ratings 
final_ratings.shape
final_ratings.drop_duplicates(['user_id','title'],inplace=True) ## droping the same record 
final_ratings.shape
book_pivot=final_ratings.pivot_table(columns='user_id',index='title',values='rating') ## pivot table
book_pivot.shape
book_pivot.fillna(0,inplace=True)
book_pivot
from scipy.sparse import csr_matrix

book_sparse=csr_matrix(book_pivot)
type(book_sparse)
from sklearn.neighbors import NearestNeighbors

model=NearestNeighbors(algorithm='brute') ## model
model.fit(book_sparse)
book_pivot.iloc[237,:].values.reshape(1,-1)
distances,suggestions=model.kneighbors(book_pivot.iloc[54,:].values.reshape(1,-1))
distances
suggestions ## recommendation
for i in range(len(suggestions)):

    print(book_pivot.index[suggestions[i]])
def reco(book_name):

    book_id=np.where(book_pivot.index==book_name)[0][0]

    distances,suggestions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1))

    

    

    

    for i in range(len(suggestions)):

        if i==0:

            print("the suggestions are ",book_name,"are : ")

        if not i:

            print(book_pivot.index[suggestions[i]])
reco('Animal Farm')