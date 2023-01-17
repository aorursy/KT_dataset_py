import numpy as np 

import pandas as pd 

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

items = pd.read_csv('../input/amazon-cell-phones-reviews/20190928-items.csv')

reviews = pd.read_csv('../input/amazon-cell-phones-reviews/20190928-reviews.csv')
items.head(2)
reviews.head(2)
items.groupby('brand')['totalReviews'].count()
px.bar(items.groupby('brand')['rating'].count().reset_index().sort_values(by='rating',ascending=False),

       x='brand', y='rating', title='Rating Count')
px.bar(items.groupby('brand')['rating'].mean().reset_index().sort_values(by='rating',ascending=False),

       x='brand', y='rating', title='Average Rating')
px.histogram(items, x='prices', title='Phone Prices')
px.scatter(items, x='prices', y='rating', title='Ratings vs Prices')
reviews.info()
reviews['date'] = pd.to_datetime(reviews.date)
reviews['year'] = reviews.date.dt.year

reviews['month'] = reviews.date.dt.month
items.asin = items.asin.astype('str')

reviews.asin = reviews.asin.astype('str')
r_items = items.merge(reviews, on='asin', suffixes=('_items', '_reviews'))
r_items.head(2)
r_items.groupby('year')['totalReviews'].sum()
r_items.groupby('month')['totalReviews'].sum().reset_index().sort_values(by='totalReviews', ascending=False)
top_phones = r_items.loc[((r_items.brand == 'Apple') | (r_items.brand == 'Samsung') | (r_items.brand == 'HUAWEI')) &

                        (r_items.year > 2012)]
px.line(r_items.loc[r_items.year > 2012].groupby(['brand', 'year'])[['rating_reviews', 'totalReviews']].sum().reset_index(),

        color='brand', x='year', y='totalReviews', title='Total Reviews By Year')
px.line(r_items.loc[r_items.year > 2012].groupby(['brand', 'year'])[['rating_reviews', 'totalReviews']].mean().reset_index(),

        color='brand', x='year', y='rating_reviews', title='Average Ratings by Year')
# split the prices to separate the two and take the first

r_items.prices = r_items.prices.str.split(',')
type([9]) == list
def return_element(x):

    if type(x) == list:

        return x[0]

    else:

        return x



r_items.prices = r_items.prices.apply(lambda x: return_element(x))
r_items.prices = r_items.prices.str[1:].astype(float)
r_items.loc[r_items.year > 2012].groupby(['brand', 'year'])[['rating_reviews', 'prices']].mean().head()
px.line(r_items.loc[r_items.year > 2012].groupby(['brand', 'year'])[['prices']].mean().reset_index(),

        color='brand', x='year', y='prices', title='Total Reviews By Year')