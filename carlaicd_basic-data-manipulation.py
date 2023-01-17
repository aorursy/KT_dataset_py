# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# load data file
df = pd.read_csv('/kaggle/input/goodreadsbooks/books.csv',index_col='bookID', error_bad_lines=False)
df.head()
df.rename(columns={'num_pages': 'pages', 'lang_code': 'lang'}, inplace=True)
df.head()
df.replace(to_replace='J.K. Rowling-Mary GrandPré', value='J.K Rowling', inplace=True)
df.head()
df['authors'].value_counts().head(15)
top_rating= df[df['average_rating'] >= 4.5]
top_rating= top_rating.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).set_index('authors')
top_rating.head(10)
top_books = df[df['average_rating'] > 4.5]
print('Top 10 books', top_books['title'].head(10))
df[df['authors'] == 'Gabriel García Márquez'].sort_values(['ratings_count','text_reviews_count'], ascending=False).head(5)
df[df['authors'] == 'Dan Brown'].sort_values(['ratings_count','text_reviews_count'], ascending=False).head(5)
df[df['authors'] == 'J.K. Rowling'].sort_values(['ratings_count','text_reviews_count'], ascending=False).head(5)
df[df['lang'] == 'spa'].sort_values(['ratings_count'], ascending=False).head(10)