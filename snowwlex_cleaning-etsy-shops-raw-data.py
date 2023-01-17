import pandas as pd

import numpy as np

import re

import datetime

import geopandas
df = pd.read_csv("../input/etsy-shops/raw_shops.csv")

df = df.drop_duplicates() # to remove duplicates in case of scrapy run several times
df.head()
df = df.drop(columns=['number_of_admirers','date_of_last_review_left'])

df.head()
def update_nan(value):

    if value == '-':

        return np.nan

    if value == '':

        return np.nan

    return value



df = df.applymap(update_nan)

df.head()
def clean_join_date(seller_join_date):

    pattern = '^On Etsy since ([0-9]+)$'

    res = re.match(pattern, seller_join_date)

    return res.group(1)

    return seller_join_date

    

def clean_number_of_sales(number_of_sales):

    pattern = '^([0-9]+) Sale'

    res = re.match(pattern, number_of_sales)

    return res.group(1)



def clean_number_of_reviews(number_of_reviews):

    pattern = '^\(([0-9]+)\)$'

    res = re.match(pattern, number_of_reviews)

    if res:

        return res.group(1)

    return np.nan



df['seller_join_date'] = df['seller_join_date'].map(clean_join_date, na_action='ignore')

df['number_of_sales'] = df['number_of_sales'].map(clean_number_of_sales, na_action='ignore')

df['number_of_reviews'] = df['number_of_reviews'].map(clean_number_of_reviews, na_action='ignore')

df.head()
df[df['average_review_score'] == '0'].head()
# 0 actually means that Etsy just does not show the it - so make it NaN as well

df.loc[ (df['average_review_score'] == '0'), 'average_review_score'] = np.nan

df.to_csv('shops.csv', index=False) #this is the same shops.csv from the input
def parse_location(location):

    splitted = location.split(",")

    if len(splitted) == 3:

        return (splitted[0].strip(), splitted[2].strip())

    elif len(splitted) == 2:

        return (splitted[0].strip(), splitted[1].strip())

    return (np.nan, splitted[0].strip())

    

def get_town(location):

    (town, country) = parse_location(location)

    return town



def get_country(location):

    (town, country) = parse_location(location)

    return country



df['seller_town'] = df['seller_location'].map(get_town, na_action='ignore')

df['seller_country'] = df['seller_location'].map(get_country, na_action='ignore')

df.head()
df.to_csv('shops_add.csv', index=False)
## prepare geocodes

wc = pd.read_csv("../input/world-cities/worldcities.csv") # taken from https://www.kaggle.com/juanmah/world-cities

wc = wc[['city', 'country', 'lat', 'lng']]

wc['index'] = wc['city'] + ", " + wc['country']

wc.head()
#get geocodes for States



# commented, as asking Nominatim takes some time

# states = geocodes[ geocodes['seller_country'] == 'United States']['seller_town'].drop_duplicates().dropna()

# states_geo = geopandas.tools.geocode(states.to_list(), provider='nominatim', user_agent="snowwlex-app")

# states_df = pd.DataFrame({

#     'state_name' : states.to_list(),

#     'lng' : states_geo['geometry'].apply(lambda p: p.x),

#     'lat' : states_geo['geometry'].apply(lambda p: p.y)

# })

# states_df.to_csv('states_geocodes.csv', index=False)



# so just load

states_df = pd.read_csv("../input/us-states-geocodes-by-nominatim-api/states_geocodes.csv")

states_df['index'] = states_df['state_name'] + ", United States"

#combine World Cities with States-specific data:

# as for US, it's state what is specified as 'city'

geo_locations = wc[ wc['country'] != 'United States' ][['index', 'lat', 'lng']].append(states_df[['index', 'lat', 'lng']])
locations = df[['seller_location', 'seller_country', 'seller_town']].drop_duplicates().dropna(subset=['seller_location'])

locations.head()
locations[ locations['seller_country'] == 'United States'].head()
#Uniting the United Kingdom

locations.loc[(locations['seller_country'] == 'England'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Wales'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Scotland'), 'seller_country'] = 'United Kingdom'

locations.loc[(locations['seller_country'] == 'Northern Ireland'), 'seller_country'] = 'United Kingdom'



locations['index'] = locations['seller_town'] + ", " + locations['seller_country']

locations.head()
locations_joined = locations.set_index('index').join(geo_locations.set_index('index'), how='inner')

locations_joined[ locations_joined['seller_country'] == 'United Kingdom'].sample(n=10)
locations_joined[ locations_joined['seller_country'] == 'United States'].sample(n=10)
locations_joined[ locations_joined['seller_country'] == 'United States']
all_locations = locations_joined.reset_index()[['seller_location', 'lng', 'lat']]

all_locations.head()
all_locations.to_csv('all_locations.csv', index=False)