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

import plotly.express as px

from iso3166 import countries
u_cols = ['user_id', 'location', 'age']

users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1', low_memory=False, skiprows=1)

b_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']

books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=b_cols, encoding='latin-1', low_memory=False, skiprows=1)

r_cols = ['user_id', 'isbn', 'rating']

ratings = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1', low_memory=False, skiprows=1)
df = pd.merge(users, ratings, on='user_id')

df = pd.merge(df, books, on='isbn')

df
ds = df['rating'].value_counts().reset_index()

ds.columns = ['value', 'count']

fig = px.bar(

    ds, 

    x='value', 

    y="count", 

    orientation='v', 

    title='Ranking distribution', 

    width=800,

    height=600

)

fig.show()
ds = df['year_of_publication'].value_counts().reset_index()

ds.columns = ['value', 'count']

ds['value'] = ds['value'] + ' year'

ds = ds.sort_values('count')

fig = px.bar(

    ds.tail(50), 

    x='count', 

    y="value", 

    orientation='h', 

    title='Top 50 years of publishing', 

    width=900,

    height=900

)

fig.show()
ds = df['book_author'].value_counts().reset_index()

ds.columns = ['author', 'count']

ds = ds.sort_values('count')

fig = px.bar(

    ds.tail(50), 

    x='count', 

    y="author", 

    orientation='h', 

    title='Authors with largest number of votes', 

    width=900,

    height=900

)

fig.show()
ds = df['book_title'].value_counts().reset_index()

ds.columns = ['book_title', 'count']

ds = ds.sort_values('count')

fig = px.bar(

    ds.tail(50), 

    x='count', 

    y='book_title', 

    orientation='h', 

    title='Books with largest number of votes', 

    width=900,

    height=900

)

fig.show()
fig = px.histogram(

    df, 

    "age", 

    nbins=100, 

    title='Age distribution', 

    width=700,

    height=600

)

fig.show()
data = df.groupby('rating')['age'].mean().reset_index()

fig = px.bar(

    data, 

    x="rating", 

    y="age", 

    orientation='v', 

    title='Average age for every raiting',

    width=800,

    height=700

)

fig.show()
users = df['user_id'].value_counts().reset_index()

users.columns = ['user_id', 'evaluation_count']

users['user_id'] = 'user ' + users['user_id'].astype(str)

users = users.sort_values('evaluation_count')

fig = px.bar(

    users.tail(50), 

    x="evaluation_count", 

    y="user_id", 

    orientation='h', 

    title='Top 50 book reviewers',

    width=800,

    height=900

)

fig.show()
users = df['user_id'].value_counts().reset_index()

users.columns = ['user_id', 'evaluation_count']

df = pd.merge(df, users)

mean_df = df[df['evaluation_count']>100]

mean_df = mean_df.groupby('user_id')['rating'].mean().reset_index().sort_values('rating')

mean_df['user_id'] = 'user ' + mean_df['user_id'].astype(str)



fig = px.bar(

    mean_df.tail(50), 

    x="rating", 

    y="user_id", 

    orientation='h', 

    title='Top 50 users with highest avarage rating (more than 100 evaluations)',

    width=800,

    height=900

)

fig.show()
books = df['book_title'].value_counts().reset_index()

books.columns = ['book_title', 'book_evaluation_count']

df = pd.merge(df, books)

mean_df = df[df['book_evaluation_count']>100]

mean_df = mean_df.groupby('book_title')['rating'].mean().reset_index().sort_values('rating')



fig = px.bar(

    mean_df.tail(50), 

    x="rating", 

    y="book_title", 

    orientation='h', 

    title='Top 50 books with highest avarage rating (more than 100 evaluations)',

    width=1000,

    height=900

)

fig.show()
books = df['publisher'].value_counts().reset_index()

books.columns = ['publisher', 'publisher_evaluation_count']

df = pd.merge(df, books)

mean_df = df[df['publisher_evaluation_count']>100]

mean_df = mean_df.groupby('publisher')['rating'].mean().reset_index().sort_values('rating')



fig = px.bar(

    mean_df.tail(50), 

    x="rating", 

    y="publisher", 

    orientation='h', 

    title='Top 50 publishers with highest avarage rating (more than 100 evaluations)',

    width=1000,

    height=900

)

fig.show()
books = df['book_author'].value_counts().reset_index()

books.columns = ['book_author', 'author_evaluation_count']

df = pd.merge(df, books)

mean_df = df[df['author_evaluation_count']>100]

mean_df = mean_df.groupby('book_author')['rating'].mean().reset_index().sort_values('rating')



fig = px.bar(

    mean_df.tail(50), 

    x="rating", 

    y="book_author", 

    orientation='h', 

    title='Top 50 authors with highest avarage rating (more than 100 evaluations)',

    width=1000,

    height=900

)

fig.show()
df['country'] = df['location'].str.split(',').str[2].str.lstrip().str.rstrip()

df['state'] = df['location'].str.split(',').str[1].str.lstrip().str.rstrip()

df['city'] = df['location'].str.split(',').str[0].str.lstrip().str.rstrip()

df = df.drop(['location', 'img_s', 'img_m', 'img_l'], axis=1)