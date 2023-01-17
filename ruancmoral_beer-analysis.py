# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_breweries = pd.read_csv('/kaggle/input/beers-breweries-and-beer-reviews/breweries.csv',index_col='id')

df_breweries.head()
df_beers = pd.read_csv('/kaggle/input/beers-breweries-and-beer-reviews/beers.csv',index_col='id')

df_beers.head()
# 2GB file

df_reviews = pd.read_csv('/kaggle/input/beers-breweries-and-beer-reviews/reviews.csv')

df_reviews.head()
#some ratings have missing values so where droped

df_reviews.dropna(subset=['look', 'smell', 'taste', 'feel', 'overall', 'score'],inplace=True)

df_reviews.head()
# remove beers with less than 5 ratings

df_reviews = df_reviews.groupby('beer_id').filter(lambda x : len(x)>5)

df_reviews = df_reviews.groupby('beer_id').mean()

df_reviews.head()
# join the mean ratings to the beers. 

df = df_beers.join(df_reviews)

# As the condidionf of a valid rating was having 5 or more ratings, some are missing here when combining

# So they were droped again.

df.dropna(subset=['look', 'smell', 'taste', 'feel', 'overall', 'score'],inplace=True)
# # df.to_csv('Datasets/beers/beer_with_review')

# df = pd.read_csv('Datasets/beers/beer_with_review')
plot = df.groupby('style')['abv'].mean().nlargest(15).plot(kind='bar', \

               title='Beer Styles with Highest Average Alcohol by Volume', ylim=(8.7,13.5) )

plot.set_ylabel('Average % Alcohol Brewed')

plot.set_xlabel('Style')
plot = df.groupby('style')['overall'].mean().nlargest(15).plot(kind='bar', 

                title='Beer Styles with Highest Overall Score',ylim=(3.9,4.2))

plot.set_ylabel('Score')

plot.set_xlabel('Style');
plot = df.groupby('style')['name'].count().nlargest(15).plot(kind='bar', \

               title='Most Brewed Beer Styles' )

plot.set_xlabel('Style');
breweries_types = []

for val in df_breweries['types']:

    try:

        # some types start with space, so it need to be removed

        types = val.split(',')

        types= [item.replace(" ", "") for item in types]

        breweries_types.extend(types)

    except AttributeError:

        pass

breweries_types = set(breweries_types)

print(len(breweries_types))

breweries_types
def split_type_brewery(val):

    try:

        if val.find(g) >-1:

            return 1

        else:

            return 0

    except AttributeError:

        return 0

for g in breweries_types:

    df_breweries[g] = df_breweries['types'].apply(split_type_brewery)

df_breweries
df_breweries[breweries_types].sum().sort_values().plot(kind='bar')
br_breweries = df_breweries[df_breweries['country'] == 'BR']

br_breweries[breweries_types].sum().sort_values().plot(kind='bar')
df_beers_active = df[df['retired'] =='f']

df_beers_retired = df[df['retired'] =='t']

df_beers_active
plot = df_beers_retired.groupby('style')['overall'].mean().nlargest(15).plot(kind='bar', 

                title='Retired Beer Styles with Highest Overall Score',ylim=(3.9,4.2))

plot.set_ylabel('Score')

plot.set_xlabel('Style');
plot = df_beers_active.groupby('style')['overall'].mean().nlargest(15).plot(kind='bar', 

                title='On Market Beer Styles with Highest Overall Score',ylim=(3.9,4.2))

plot.set_ylabel('Score')

plot.set_xlabel('Style');