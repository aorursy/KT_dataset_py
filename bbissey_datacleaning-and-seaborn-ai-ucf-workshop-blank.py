"""
want to work on this locally?

1. make conda environment, 
                                    conda create --name workshop
                                    conda activate workshop



2. install dependencies wiithin conda-env
                                    pip install kaggle
                                    conda install folium 
                                    conda install seaborn
                                    pip install descartes
                                    kaggle datasets download -d bbissey/barcelonaairbnbgeojson
                                    or just download the dataset on kaggle's website


^^ maybe some more package installs 
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline



df_0 = pd.read_csv('../input/barcelonaairbnbgeojson/listings.csv')




# trouble with only displaying some columns or rows?
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# scraped on sept19, 2019 from insideairbnb.com


#Remove Majority Null Columns



# Remove Columns with One Unique Value
# Why? For example, country. If all data is in the same country, we don't need 20404 rows that say Spain
# Yes, there are faster and more efficient ways to do this.


# Reassign to df_2 variable


# what data types make up our dataframe?
# object == string


# As previously seen, pandas' nunique function counts the number of unique values in a column


# Bulk Removal of Redundant Columns and String Columns

# Some of these string columns may be helpful for categorical analysis and language processing, 
# but for the purpose of this workshop we will leave them out.


# forgive the giant chunk of text. it will make sense as we fill this workshop out.
# You know I'm not typing all this out and you shouldn't either. 

'''
df_3 = df_2.drop(['listing_url', 'last_scraped', 'name', 'summary', 'space', 'description',
                 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'picture_url',
                 'host_url', 'host_name', 'host_since', 'host_location', 'host_about', 'host_thumbnail_url', 
                  'host_neighbourhood', 'host_listings_count', 'host_total_listings_count', 'host_response_time', 
                  'host_response_rate', 'street', 'host_verifications', 'host_picture_url', 'amenities', 'calendar_updated', 
                  'calendar_last_scraped', 'availability_30', 'availability_90', 'availability_60','neighbourhood', 
                  'smart_location', 'is_location_exact', 'first_review', 'last_review', 'license','minimum_minimum_nights',
                  'maximum_minimum_nights','minimum_maximum_nights','maximum_maximum_nights','minimum_nights_avg_ntm',
                  'maximum_nights_avg_ntm','calculated_host_listings_count_entire_homes',
                  'calculated_host_listings_count_shared_rooms','calculated_host_listings_count_private_rooms'], axis =1)
print(f'Before removing redundant values: {df_2.shape}')
print(f'After removing redundant values:  {df_3.shape}')

'''

# Look at what data remains, along with how many unique values there are

# set index by id

# display data again


# Why? To visualize the proportions of data relative to each other
# Normalization of value counts helps with seeing proportions easier





        

# Almost everybody has a real bed, let's remove this variable


# please don't type this all out
'''
df_4 = df_3.drop(['bed_type', 
                  'review_scores_accuracy',
                  'review_scores_cleanliness',
                  'review_scores_checkin',
                  'review_scores_communication',
                  'review_scores_location',
                  'review_scores_value',
                  'state', 
                  'zipcode',
                  'market',
], axis = 1)
'''

# check nulls again

# Returns every row where the 'availability_365' column equals 0.

# Removal of Inactive Listings


#Parsing Floats from Price Columns



# Review the remaining data




#PRICE LIMIT SETTING AND PRICE DISTRIBUTION

# Explain the parts of the violin chart:
# Curve = Normal Disribution
# Bottom of veritcal line = Smallest value
# Top of vertical line = Highest value
# Bottom of the middle box = First quartile
# Top of the middle box = Third quartile
# White dot = mean



#sns.despine

# Explain skew




#Predictor variables and Skewed Distribution


# BINNING VARIABLES (drop beds because we have bedrooms)




#log transforming price, sec deposit, cleaning fee, extra people, reviews_per_month variables

'''

df_10['log_price'] = np.log(df_10['price']*10 + 1)
df_10['log_security'] = np.log(df_10['security_deposit'] + 1)
df_10['log_cleaning'] = np.log(df_10['cleaning_fee']*10 + 1)

df_10['log_extra_people'] = np.log(df_10['extra_people']*10 + 1)
df_10['log_reviews_pm'] = np.log((df_10['reviews_per_month']+ 1) * 10)
df_10['log_bedrooms'] = np.log(df_10['bedrooms'] + 1)
df_10['log_bathrooms'] = np.log(df_10['bathrooms']*10 + 1)
df_10['log_guests_included'] = np.log(df_10['guests_included']*100 + 1)
df_10['log_listings_count'] = np.log(df_10['calculated_host_listings_count']*10 + 1)

'''




# Predictor variables and Skewed Distribution


# Predictor variables and Skewed Distribution







import descartes
import geopandas as gpd

gjsonFile = "../input/barcelonaairbnbgeojson/neighbourhoods.geojson"
barc_hoods = gpd.read_file(gjsonFile)

barc_hoods.plot(figsize=(10,10), column="neighbourhood_group", cmap = "tab10")
barc_hoods['neighbourhood_group'].value_counts()
barc_hoods.plot(figsize=(20,20), column="neighbourhood_group", cmap='tab10', alpha = .5)
#plt.figure(figsize=(20,20))

sns.scatterplot(x='longitude', 
                y = 'latitude', 
                hue='price', 
                size = 'price', 
                sizes= (20, 600),
                alpha = .8,
                marker=".",
                data = df_10,
                )
barc_hoods.plot(figsize=(10,10), alpha = .5)
#plt.figure(figsize=(20,20))
#here we add lat and longitude lines to plot for context


sns.set({'font.size' : 10})
sns.set_style("whitegrid")
#plot by reviews per month
sns.scatterplot(x='longitude', 
                y = 'latitude', 
                hue='neighbourhood_group_cleansed', 
                alpha = .5,
                marker="o",
                data = df_10,
                cmap='Set3')
                #size = 15)
import folium
from folium.plugins import HeatMap, MarkerCluster

mapp = folium.Map(location=[41.40,2.15], zoom_start=12, figsize=(20,20))
cluster = MarkerCluster().add_to(mapp)
# add a marker for every record in the filtered data, use a clustered view
for each in df_10.iterrows():
    folium.Marker(
        location = [each[1]['latitude'],each[1]['longitude']], 
        clustered_marker = True).add_to(cluster)
  
mapp.save(outfile='map.html')
display(mapp)


max_price_map = df_10['price'].max() #this should be 650
barc_map = folium.Map(location=[41.40, 2.15], zoom_start=12, )

heatmap = HeatMap( list(zip(df_10.latitude, df_10.longitude, df_10.price)),
                 min_opacity = .3,
                 max_val = max_price_map, 
                 radius = 3,
                 blur = 2,
                 max_zoom=1)

folium.GeoJson(barc_hoods).add_to(barc_map)
barc_map.add_child(heatmap)

barc_map.save(outfile="mapp.html")

import folium
from folium.plugins import HeatMap


max_reviews_pm = df_10['reviews_per_month'].max() 
barc_map = folium.Map(location=[41.40, 2.15], zoom_start=12, )

heatmap = HeatMap( list(zip(df_10.latitude, df_10.longitude, df_10.reviews_per_month)),
                 min_opacity = .3,
                 max_val = max_reviews_pm, 
                 radius = 3,
                 blur = 2,
                 max_zoom=1)

folium.GeoJson(barc_hoods).add_to(barc_map)
barc_map.add_child(heatmap)




# UGLY, but potentially helpful plot

# why may we want to do a scatterplot for categorical data??


#appears that reviews_per_month is an important activity heuristic, 
#we can see there is a potential relationship between price and bedrooms, especially in active airbnbs


# Removing Log Transformed Variables
'''
df_lg = df_10.drop(['price', 
                    'security_deposit', 
                    'cleaning_fee', 
                    'extra_people', 
                    'reviews_per_month', 
                    'bedrooms', 
                    'bathrooms', 
                    'guests_included', 
                    'calculated_host_listings_count'], axis = 1)

'''

# Remove non-integer variables that we will not be dummifying

# MODELING STARTS

                          
