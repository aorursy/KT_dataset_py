# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from matplotlib import pyplot as plt

from matplotlib import rcParams as rcP
df = pd.read_csv('/kaggle/input/pune-house-data/Pune_House_Data.csv')

df.head()
# Exploring the dataset

df.shape
# Exploring the dataset

df.groupby('area_type')['area_type'].agg('count')
# Exploring the dataset

df.groupby('availability')['availability'].agg('count')
# Exploring the dataset

df.groupby('size')['size'].agg('count')
# Exploring the dataset

df.groupby('site_location')['site_location'].agg('count')
# Removing the columns of society

df = df.drop('society', axis='columns')

df.head()
# Data Cleaning

# Checking the null values in the dataset

df.isnull().sum()
# Applying median to the balcony and bath column

from math import floor



balcony_median = float(floor(df.balcony.median()))

bath_median = float(floor(df.bath.median()))



df.balcony = df.balcony.fillna(balcony_median)

df.bath = df.bath.fillna(bath_median)



# Checking the null values in the dataset again

df.isnull().sum()
# Dropping the rows with null values because the dataset is huge as compared to null values.

df = df.dropna()

df.isnull().sum()
# Converting the size column to bhk

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

df = df.drop('size', axis='columns')

df.groupby('bhk')['bhk'].agg('count')


# Since the total_sqft contains range values such as 1133-1384, lets filter out these values

def isFloat(x):

    try:

        float(x)

    except:

        return False

    return True



# Displaying all the rows that are not integers

df[~df['total_sqft'].apply(isFloat)]
# Converting the range values to integer values and removing other types of error

def convert_sqft_to_num(x):

    tokens = x.split('-')

    if len(tokens) == 2:

        return (float(tokens[0])+float(tokens[1]))/2

    try:

        return float(x)

    except:

        return None

    

df['new_total_sqft'] = df.total_sqft.apply(convert_sqft_to_num)

df = df.drop('total_sqft', axis='columns')

df.head()
# Removing the rows in new_total_sqft column that hase None values

df.isna().sum()
df = df.dropna()

df.isnull().sum()
# Adding a new column of price_per_sqft

df1 = df.copy()



# In our dataset the price column is in Lakhs

df1['price_per_sqft'] = (df1['price']*100000)/df1['new_total_sqft']

df1.head()
# Checking unique values of 'location' column

locations = list(df['site_location'].unique())

print(len(locations))
# Removing the extra spaces at the end

df1.site_location = df1.site_location.apply(lambda x: x.strip())



# Calulating all the unqiue values in 'site_location' column

location_stats = df1.groupby('site_location')['site_location'].agg('count').sort_values(ascending=False)

location_stats


# Checking locations with less than 10 values

print(len(location_stats[location_stats<=10]), len(df1.site_location.unique()))
# Labelling the locations with less than or equal to 10 occurences to 'other'

locations_less_than_10 = location_stats[location_stats<=10]



df1.site_location = df1.site_location.apply(lambda x: 'other' if x in locations_less_than_10 else x)

len(df1.site_location.unique())


# Checking the unique values in 'availability column'

df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)
# Labelling the dates into Not Ready

dates = df1.groupby('availability')['availability'].agg('count').sort_values(ascending=False)



dates_not_ready = dates[dates<10000]

df1.availability = df1.availability.apply(lambda x: 'Not Ready' if x in dates_not_ready else x)



len(df1.availability.unique())
# Checking the unique values in 'area_type' column

df1.groupby('area_type')['area_type'].agg('count').sort_values(ascending=False)



# Since the column has only few unique values, we don't perform any operation