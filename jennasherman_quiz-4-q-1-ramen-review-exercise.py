# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ramen = pd.read_csv('/kaggle/input/ramen-ratings/ramen-ratings.csv')

ramen = ramen.set_index('Review #')

ramen.head()
ramen['Stars']=pd.to_numeric(ramen['Stars'],errors='coerce')
ramen.shape #shows how many rows and how many columns in the data set
brands = ramen.groupby('Brand')

reviews = brands.mean().rename(columns = {'Stars': 'Average_Stars'})

reviews['review_count'] = brands.size()

reviews.head()
top_reviews = reviews.review_count.sort_values(ascending = False).iloc[0:16]

top_reviews.head(15)
high_reviews = reviews[reviews.Average_Stars >= 4.5]

high_reviews = high_reviews[high_reviews.review_count >= 2]

high_reviews.head(10)
high_reviews.shape
reviews.Average_Stars.sort_values(ascending = False).tail(10)
country = ramen.groupby('Country')

country_ratings = country.mean().rename(columns = {'Stars': 'Average_Country_Stars'})

max_rating = country_ratings.Average_Country_Stars.max()

max_country = country_ratings.Average_Country_Stars.idxmax()

print('The country with the most reviews is ' + max_country + ' with the average star rating of ' + str(max_rating) + '.')
country_ratings['total_reviews'] = country.size()

most_reviews = country_ratings.total_reviews.sort_values(ascending = False).reset_index().iloc[0]

print(most_reviews)

country_ratings.loc[most_reviews.Country] 
style = ramen.groupby('Style').size()

style.plot.pie()