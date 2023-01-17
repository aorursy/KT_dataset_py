import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re

import matplotlib.style as style

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
books = pd.read_csv("/kaggle/input/goodbooks-10k/books.csv")

book_tags = pd.read_csv("/kaggle/input/goodbooks-10k/book_tags.csv")

tags = pd.read_csv("/kaggle/input/goodbooks-10k/tags.csv")

ratings = pd.read_csv("/kaggle/input/goodbooks-10k/ratings.csv")
tags.head()
book_tags.head()
#Left join between book_tags and tags dataframe

book_tags = pd.merge(book_tags,tags,on='tag_id',how='left')
book_tags.drop(book_tags[book_tags.duplicated()].index, inplace = True)
book_tags
books.head()
#Drop unnecessary columns

books.drop(columns=['id', 'best_book_id', 'work_id', 'isbn', 'isbn13', 'title','work_ratings_count','ratings_count','work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3','ratings_4', 'ratings_5', 'image_url','small_image_url'], inplace= True)



#Rename columns

books.rename(columns={'original_publication_year':'pub_year', 'original_title':'title', 'language_code':'language', 'average_rating':'rating'}, inplace=True)
books.isnull().sum()
#Dropping the null values

books.dropna(inplace= True)
#Using python's split string function to create a list of authors

books['authors'] = books.authors.str.split(',')
books
book_authors = books.copy()



#For every row in the dataframe, iterate through the list of authors and place a 1 into the corresponding column

for index, row in books.iterrows():

    for author in row['authors']:

        book_authors.at[index, author] = 1

        

#Filling in the NaN values with 0 to show that a book isn't written by that author

book_authors = book_authors.fillna(0)

book_authors.head()
#Generalising the format of author names for simplicity in future

book_authors.columns = [c.lower().strip().replace(' ', '_') for c in book_authors.columns]



#Setting book_id as index of the dataframe 

book_authors = book_authors.set_index(book_authors['book_id'])



#Dropping unnecessary columns

book_authors.drop(columns= {'book_id','pub_year','title','rating','books_count', 'authors','language'}, inplace=True)
book_authors.head()
user_1 = pd.DataFrame([{'book_id':2767052, 'rating':5.0},{'book_id':3, 'rating':4.0}, {'book_id':41865, 'rating':4.5},{'book_id':15613, 'rating':3.0},{'book_id':2657, 'rating':2.5}])

user_1
user_authors = book_authors[book_authors.index.isin(user_1['book_id'].tolist())].reset_index(drop=True)

user_authors
user_1.rating
#Dot product to get weights

userProfile = user_authors.transpose().dot(user_1['rating'])

#The user profile

userProfile
recommendation = (((book_authors*userProfile).sum(axis=1))/(userProfile.sum())).sort_values(ascending=False)

#Top 20 recommendations

recommendation.head(20)
#The final recommendation table

books.loc[books['book_id'].isin(recommendation.head(20).keys())].reset_index()