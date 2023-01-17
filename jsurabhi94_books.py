# importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings  

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set(rc={'figure.figsize':(12,8)})
books = pd.read_csv('../input/books.csv', error_bad_lines = False, index_col = 'bookID')

books.head()
print (books.shape)

print()

books.info()
# datatype present in the dataset

books.dtypes
# converting the datatype of the 'isbn13' column

books['isbn13'] = books['isbn13'].astype('object')
books['authors'].replace(to_replace = 'J.K. Rowling-Mary GrandPré', value = 'J.K. Rowling', inplace = True)

books.rename(columns = {'# num_pages' : 'Total_pages'}, inplace = True)



# droping the columns with the isbn numbers

books.drop(['isbn', 'isbn13'], axis = 1, inplace = True)



books.head(5)
books.language_code.value_counts()
sns.countplot(x = 'language_code', data = books)
books.authors.value_counts().head(10)
books.authors.value_counts().sort_values(ascending = False).head(10).plot.bar(figsize = (12,8), colormap = 'Accent', rot = 45, yticks = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70])
rated_10 = books.sort_values('ratings_count', ascending = False).set_index('title').head(10)

rated_10.ratings_count.plot(kind = 'bar', colormap = 'rainbow', figsize = (12,8))
sns.distplot(books.average_rating,color = 'g')
pages_10 = books.sort_values(by = 'Total_pages', ascending = False). head(10).set_index('title')

pages_10.Total_pages.plot(kind = 'barh', figsize = (8,10))
text_10 = books.sort_values(by = 'text_reviews_count', ascending = False).set_index('title').head(10)

text_10.text_reviews_count.plot.bar(colormap = 'icefire_r')
text = books[['title', 'authors', 'ratings_count', 'text_reviews_count']]

text = text.groupby(['authors'])['text_reviews_count'].sum().sort_values(ascending = False).reset_index()

text.head(10)
rating = books.groupby('authors')['ratings_count'].sum().sort_values(ascending = False).reset_index()

rating.head()