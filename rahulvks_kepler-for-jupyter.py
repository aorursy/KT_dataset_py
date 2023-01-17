# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keplergl import KeplerGl



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
listings = pd.read_csv("../input/listings.csv",nrows=1000)

listings_details = pd.read_csv("../input/listings_details.csv", low_memory=True,nrows=1000)

print(listings.shape)

target_columns = ["id", "property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "street", "weekly_price", "monthly_price", "market"]

listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')

listings.info()

map_1 = KeplerGl()

map_1
listings['price'].describe()
listings['price'].hist(bins=25, grid=False)
price_above_200 = listings[listings['price'] > 200]

col = ['latitude','longitude','neighbourhood','price','market']

loccation_map = price_above_200[col]
loccation_map.head()
map_1.add_data(data=loccation_map, name='data_1')

map_1
accommodations_2_person = listings[listings['accommodates']==2]

loccation_map = accommodations_2_person[col]
map_2 = KeplerGl()

map_2.add_data(data=loccation_map, name='data_1')

map_2