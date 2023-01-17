# import pandas

import pandas as pd



# creating a DataFrame

pd.DataFrame({'Yes': [50, 31], 'No': [101, 2]})
# another example of creating a dataframe

pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 'Sue': ['Pretty good.', 'Bland']})
pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 

              'Sue': ['Pretty good.', 'Bland.']},

              index = ['Product A', 'Product B'])
# creating a pandas series

pd.Series([1, 2, 3, 4, 5])
# we can think of a Series as a column of a DataFrame.

# we can assign index values to Series in same way as pandas DataFrame

pd.Series([10, 20, 30], index=['2015 sales', '2016 sales', '2017 sales'], name='Product A')
import os

os.listdir("../input/188-million-us-wildfires")
# reading a csv file and storing it in a variable

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv")
# we can use the 'shape' attribute to check size of dataset

wine_reviews.shape
# To show first five rows of data, use 'head()' method

wine_reviews.head()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

wine_reviews.head()
import sqlite3

conn = sqlite3.connect("../input/188-million-us-wildfires/FPA_FOD_20170508.sqlite")
fires = pd.read_sql_query("SELECT * FROM fires", conn)
fires.head()
wine_reviews.head().to_csv("wine_reviews.csv")
conn = sqlite3.connect("fires.sqlite")

fires.head(10).to_sql("fires", conn)
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
reviews
# access 'country' property (or column) of 'reviews' 

reviews.country
# Another way to do above operation

# when a column name contains space, we have to use this method

reviews['country']
# To access first row of country column

reviews['country'][0]
# returns first row

reviews.iloc[0]
# returns first column (country) (all rows due to ':')

reviews.iloc[:, 0]
# retruns first 3 rows of first column

reviews.iloc[:3, 0]
# we can pass a list of indices of rows/columns to select

reviews.iloc[[0, 1, 2, 3], 0]
# We can also pass negative numbers as we do in Python

reviews.iloc[-5:]
# To select first entry in country column

reviews.loc[0, 'country']
# select columns by name using 'loc'

reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
# 'set_index' to the 'title' field

reviews.set_index('title')
# 1. Find out whether wine is produced in Italy

reviews.country == 'Italy'
# 2. Now select all wines produced in Italy

reviews.loc[reviews.country == 'Italy'] #reviews[reviews.country == 'Italy']
# Add one more condition for points to find better than average wines produced in Italy

reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]  # use | for 'OR' condition
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
reviews['critic'] = 'everyone'

reviews.critic
# using iterable for assigning

reviews['index_backwards'] = range(len(reviews), 0, -1)

reviews['index_backwards']
import pandas as pd

pd.set_option('max_rows', 5)

import numpy as np

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews.head()
reviews.describe()

# this method generates stats for numerical data only
reviews.taster_name.describe()          

# when 'describe' method is applied to string data
# Find out a particular statistic of a DataFrame or Series

# For eg. find the average of points/rating given to wines

reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()
review_points_mean = reviews.points.mean()

reviews.points.map(lambda p: p - review_points_mean)
def remean_points(row):

    row.points = row.points - review_points_mean

    return row



reviews.apply(remean_points, axis='columns')
reviews.head(1)
# Another way (also faster one) to remean points

review_points_mean = reviews.points.mean()

reviews.points - review_points_mean
# Combining data from two string columns. concatenation

reviews.country + ' - ' + reviews.region_1
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
reviews.groupby('points').points.count()
reviews.groupby('points').count()
reviews.groupby('points').price.min()
reviews.groupby('winery').apply(lambda df: df.title.iloc[0])
# reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.argmax()])

reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
reviews.groupby('country').price.agg([len, min, max])
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])

countries_reviewed
mi = _.index

type(mi)
countries_reviewed.reset_index()
countries_reviewed = countries_reviewed.reset_index()

countries_reviewed.sort_values(by='len')
# Descending sort

countries_reviewed.sort_values(by='len', ascending=False)
# sort by index

countries_reviewed.sort_index()
# sort by more than one column at a time

countries_reviewed.sort_values(by=['country', 'len'])
import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option('max_rows', 5)
reviews.price.dtype
# data types of all columns in a DataFrame

reviews.dtypes
reviews.points.astype('float64')
# data type of index of "Series" or "DataFrame"

reviews.index.dtype
reviews[reviews.country.isnull()]
reviews.region_2.fillna("Unknown")
reviews.taster_twitter_handle.replace('@kerinokeefe', '@kerino')
import pandas as pd

pd.set_option('max_rows', 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews
reviews.rename(columns={'points': 'score'})
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
reviews.rename_axis('fields', axis='columns').rename_axis('wines', axis='rows')



#reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')    # Also correct
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")

british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")



pd.concat([canadian_youtube, british_youtube])
left = canadian_youtube.set_index(['title', 'trending_date'])

right = british_youtube.set_index(['title', 'trending_date'])



left.join(right, lsuffix='_CAN', rsuffix='_UK')