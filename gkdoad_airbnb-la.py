# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/listings.csv', low_memory=False)

data.head()
data.shape
data.info()
data.columns
print(data.dtypes.head(50))

print(data.dtypes.tail(56))
# Creating a copy of data

df = data
# Checking null values in each column

print(df.isnull().sum().head(50))

print(df.isnull().sum().tail(56))
# dropping columns containing mostly null values as it won't help predicting price

drop_columns = ['thumbnail_url','medium_url','xl_picture_url','host_acceptance_rate','neighbourhood_group_cleansed','square_feet','license']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
# Checking remaining columns

print(df.dtypes.head(50))

print(df.dtypes.tail(56))
# A quick look at listing.csv reveals that some columns won't help predicting price, so dropping those columns

drop_columns = ['listing_url','scrape_id','last_scraped','name','summary','space','description','neighborhood_overview','notes',

                'transit','access','interaction','house_rules','picture_url','host_id','host_url','host_name','host_location',

                'host_about','host_thumbnail_url','host_picture_url','host_verifications','calendar_last_scraped','jurisdiction_names']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
# since all listings are only LA based to country_code, and country will be same throughout the data, these columns can be dropped

drop_columns = ['country_code','country']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
# Taking a quick look at data with numeric values

df.hist(figsize=(100,100))
# Lets check values for columns showing only single line

columns = ['has_availability','host_has_profile_pic','is_business_travel_ready','require_guest_phone_verification',

           'require_guest_profile_picture', 'requires_license']

for column in columns:

    print(df[column].value_counts())
# Above 2 checks show these columns won't be helpful predicting price, so drop these columns

df.drop(columns, axis=1, inplace=True)

df.shape
# Converting monet columns to numeric values from string to make them more useful

df['price'] = df['price'].str[1:-3]

df['price'] = df['price'].str.replace(",", "")

df['price'].fillna(0, inplace=True)

df['price'] = df['price'].astype('int64')



df['security_deposit'] = df['security_deposit'].str[1:-3]

df['security_deposit'] = df['security_deposit'].str.replace(",", "")

df['security_deposit'].fillna(0, inplace=True)

df['security_deposit'] = df['security_deposit'].astype('int64')



df['cleaning_fee'] = df['cleaning_fee'].str[1:-3]

df['cleaning_fee'] = df['cleaning_fee'].str.replace(",", "")

df['cleaning_fee'].fillna(0, inplace=True)

df['cleaning_fee'] = df['cleaning_fee'].astype('int64')



df['extra_people'] = df['extra_people'].str[1:-3]

df['extra_people'] = df['extra_people'].str.replace(",", "")

df['extra_people'].fillna(0, inplace=True)

df['extra_people'] = df['extra_people'].astype('int64')



df[['price','security_deposit','cleaning_fee','extra_people']].head()
# Prining remaining columns

print(df.dtypes.head(30))

print(df.dtypes.tail(37))
# neighbourhood and neighbourhood_cleansed columns have same values. neighbourhood_cleansed is filled more accurately.

# drop neighbourhood column

df.drop(['neighbourhood'], axis=1, inplace=True)

df.shape
# Some date columns seem to add no value and can be dropped

drop_columns = ['host_since','first_review','last_review']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
# Converting true false columns to boolean values t = 1, f = 0

df.replace({'t': 1,'f': 0, }, inplace=True)

df.head()
# Working only on neighboorhood_cleansed columns can yield same results as many other combines, so I can drop these columns

drop_columns = ['zipcode','latitude','longitude','street','city','state','market','smart_location','is_location_exact']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
# Prining remaining columns

print(df.dtypes.head(30))

print(df.dtypes.tail(24))
# Drop any duplicate records

df.drop_duplicates(inplace=True)

df.shape
# There are many columns minimum_nights maximum_nights, minimum_minimum_nights, maximum_minimum_nights, minimum_maximum_nights,

# maximum_maximum_nights, minimum_nights_avg_ntm, maximum_nights_avg_ntm. Keeping only minimum_nights, maximum_nights

drop_columns = ['minimum_minimum_nights','maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights',

                'minimum_nights_avg_ntm','maximum_nights_avg_ntm']

df.drop(drop_columns, axis=1, inplace=True)

df.shape
df.dtypes.head(48)