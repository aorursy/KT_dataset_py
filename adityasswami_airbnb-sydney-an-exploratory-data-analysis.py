# Import libraries and packages



import numpy as np

import pandas as pd

from pandas.api.types import is_string_dtype

from pandas.api.types import is_numeric_dtype

import seaborn as sns

import matplotlib.pyplot as plt

import csv

import json

import geojson

import folium

from folium import plugins

from folium.plugins import FastMarkerCluster

from folium.plugins import TimestampedGeoJson

import datetime

import geopandas as gpd

import geoplot

import geoplot.crs as gcrs

import geoplot as gplt



%matplotlib inline
# Import listings data, load as pandas dataframe

listings_data = pd.read_csv('../input/listings_march2019.csv')



# Import reviews data, load as a pandas dataframe

reviews_data = pd.read_csv('../input/reviews_march2019.csv')
# Load the geojson file required for mapping as a geodataframe

syd_geo = gpd.GeoDataFrame.from_file('../input/neighbourhoods.geojson')
# Create dataframe listings with relevant columns



listings = listings_data[['id','neighbourhood','neighbourhood_cleansed','latitude','longitude','property_type','room_type','accommodates','price','host_since',

    'guests_included','minimum_nights','number_of_reviews','review_scores_rating','instant_bookable','cancellation_policy']]
# Price is in string format with '$'' and ','' Remove and convert to float

listings.price = listings.price.replace('[\$,]', '', regex=True).astype(float)
# Convert host_since from string to datetime

listings['host_since'] = pd.to_datetime(listings['host_since'])
# Create a series to group listings by year and convert to a dataframe

result = listings.groupby(listings['host_since'].map(lambda x: x.year)).id.count()

year = pd.DataFrame(result)
# Rename column id to listings and reset index

year.rename(columns={'id':'listings'}, inplace=True)

year = pd.DataFrame(year).reset_index()
# Create a new column with the cumulative listings by year

year['total listings']= year['listings'].cumsum(axis=0)
# Convert host_since data type to int

year['host_since'] = year['host_since'].astype(int)
# Rename column host_since to year

year.rename(columns={'host_since':'year'}, inplace=True)
# Plot Airbnb listings over time

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x="year", y="total listings", data=year, color='Coral')

ax.set_title('Airbnb Listings in Sydney',fontsize=25,pad=20) # Give the plot a main title

ax.set_xlabel('Year',fontsize=15, labelpad=15) # Set text for the x axis,

ax.set_ylabel('Number of Listings',fontsize=15, labelpad=15)# Set text for y axis  

sns.despine(offset=5, left=True)
# Plot new Airbnb listings over time

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x='year', y="listings", data=year, color="coral")

ax.set_title('New Airbnb Listings, Sydney',fontsize=20,pad=20) # Give the plot a main title

ax.set_xlabel('Year',fontsize=14, labelpad=15) # Set text for the x axis,

ax.set_ylabel('Listings',fontsize=14, labelpad=15)# Set text for y axis  

sns.despine(offset=5, left=True)
# Plot the room types in Sydney

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax.axes.set_title("Room Type, Airbnb Sydney",fontsize=20, pad=20)

ax = sns.countplot(y='room_type',data=listings,order=listings['room_type'].value_counts().index, palette="Set3")

ax.set_xlabel('Number of Listings',fontsize=14,labelpad=15)

ax.set_ylabel('Type of Room',fontsize=14,labelpad=15)

ax.xaxis.set_tick_params(labelsize=10)

ax.yaxis.set_tick_params(labelsize=10)
# Plot top 20 neighbourhoods in terms listings

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(25, 15)

ax.axes.set_title("Most Popular Neighourhoods",fontsize=40,pad=40)

ax = sns.countplot(y='neighbourhood',data=listings, order = listings['neighbourhood'].value_counts().iloc[:20].index, palette="Set3")

ax.set_xlabel('Number of Listings',fontsize=30,labelpad=30)

ax.set_ylabel('Neighbourhood',fontsize=30,labelpad=30)

ax.xaxis.set_tick_params(labelsize=20)

ax.yaxis.set_tick_params(labelsize=20)
# Create a list of top 20 neighbourhoords in terms of listings

top = listings['neighbourhood'].value_counts().iloc[:20].index.tolist()
# Create a dataframe to group neighbourhoods by average room price

price = pd.DataFrame(listings.groupby(['neighbourhood']).price.mean().reset_index())



# Create a dataframe to filter top 20 neighbourhoods

top_price = price[price['neighbourhood'].isin(top)].sort_values('price',ascending=False)
# Barplot of price by neighbourhood, top 20 neighbourhoods in terms of listings

fig, ax = plt.subplots()

fig.set_size_inches(25, 15)

ax.axes.set_title("Room Price, Airbnb Sydney",fontsize=40, pad=40)

ax = sns.barplot(x='price', y='neighbourhood',data=top_price, palette='Set3')

ax.set_xlabel('Avg. Price',fontsize=30,labelpad=30)

ax.set_ylabel('Neighbourhood',fontsize=30,labelpad=30)

ax.xaxis.set_tick_params(labelsize=20)

ax.yaxis.set_tick_params(labelsize=20)
# Boxplot of the ratings for the listings

fig, ax = plt.subplots(figsize=(20,10))



ax = sns.boxplot(x=listings["review_scores_rating"], linewidth=1, palette='Set2')

ax.set_title('Avg. Rating, Airbnb Sydney',fontsize=30,pad=30) # Give the plot a main title

ax.set_xlabel('Rating',fontsize=20, labelpad=15) # Set text for the x axis,



sns.despine(offset=5, left=True)
# Rename column from date to review_date in the reviews data

reviews_data.rename(columns={'date':'review_date'}, 

                 inplace=True)
# Change data type to datetime

reviews_data['review_date'] = pd.to_datetime(reviews_data['review_date'])
# Create new columns for year, month and month year

reviews_data['month_year'] = reviews_data.review_date.dt.to_period('M')

reviews_data['year'] = reviews_data.review_date.apply(lambda x: x.year)

reviews_data['month'] = reviews_data.review_date.apply(lambda x: x.month)
# Create new dataframe to group reviews by year

reviews_year = pd.DataFrame(reviews_data.groupby(['year']).review_date.count().reset_index())
# Rename column from review_date to reviews

reviews_year.rename(columns={'review_date':'reviews'}, 

                 inplace=True)
# Create new column for bookings, estimate bookings by mupltiplying reviews by 2

reviews_year['bookings'] = reviews_year['reviews']*2
# Plot bookings per year

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x="year", y="bookings", data=reviews_year, color='Coral')

ax.set_title('Bookings per Year, Airbnb Sydney',fontsize=20,pad=20) # Give the plot a main title

ax.set_xlabel('Year',fontsize=15, labelpad=15) # Set text for the x axis,

ax.set_ylabel('Estimated Bookings',fontsize=15, labelpad=15)# Set text for y axis  

sns.despine(offset=5, left=True)
# Merge reviews per year and listings per year

demand = year.merge(reviews_year, on='year')
# Create new columns for booked nights and available nights

demand['booked nights'] = demand['bookings']*2

demand['available nights'] = demand['total listings']*365
# Create new feature occupancy rate, booked nights divided by available nights

demand['occupancy rate'] = demand['booked nights']/demand['available nights']*100
# Plot occupancy rate

sns.set(style="whitegrid")

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x="year", y="occupancy rate", data=demand, color='Coral')

ax.set_title('Occupancy Rate, Airbnb Sydney',fontsize=20,pad=20) # Give the plot a main title

ax.set_xlabel('Year',fontsize=15, labelpad=15) # Set text for the x axis,

ax.set_ylabel('Occupancy Rate %',fontsize=15, labelpad=15)# Set text for y axis  

sns.despine(offset=5, left=True)
# Create new dataframe for listings grouped by neighbourhood

listings_geo = pd.DataFrame(listings.groupby(['neighbourhood_cleansed']).id.count()).reset_index()



# Rename column in neighbourhood_cleansed dataframe

listings_geo.rename(columns={'neighbourhood_cleansed':'neighbourhood','id':'listings'}, 

                 inplace=True)
# Create new dataframe for listings grouped by neighbourhood

price_geo = pd.DataFrame(listings.groupby(['neighbourhood_cleansed']).price.mean()).reset_index()



# Rename column in neighbourhood_cleansed dataframe

price_geo.rename(columns={'neighbourhood_cleansed':'neighbourhood','id':'listings'}, 

                 inplace=True)
# Create a new dataframe for ratings grouped by neighbourhood

ratings_geo = pd.DataFrame(listings.groupby(['neighbourhood_cleansed']).review_scores_rating.mean()).reset_index()



# Rename column in neighbourhood_cleansed dataframe

ratings_geo.rename(columns={'neighbourhood_cleansed':'neighbourhood','id':'listings'}, 

                 inplace=True)
# Drop the column neighbourhood_group, no useful data

syd_geo.drop(columns=['neighbourhood_group'],inplace=True)
# Merge listings, price and ratings with geo

syd_geo = syd_geo.merge(listings_geo, on='neighbourhood')

syd_geo = syd_geo.merge(price_geo, on='neighbourhood')

syd_geo = syd_geo.merge(ratings_geo, on='neighbourhood')
# Inspect the dataframe

syd_geo.head()
# Plot the data, Listings Choropleth



gplt.choropleth(syd_geo, hue=syd_geo['listings'], projection=gcrs.PlateCarree(),

                cmap='Blues', linewidth=0.5, edgecolor='white', k=None, legend=True, figsize=(10, 10))

plt.title("Listings per Neighbourhood, Airbnb Sydney", fontsize=20, pad=60)

plt.savefig("airbnb_listings_suburb.png", bbox_inches='tight', pad_inches=0.1)
# Plot the data, Avg. Listing Price Choropleth



gplt.choropleth(syd_geo, hue=syd_geo['price'], projection=gcrs.PlateCarree(),

                cmap='Blues', linewidth=0.5, edgecolor='white', k=None, legend=True, figsize=(10, 10))

plt.title("Avg. Listing Price per Neighbourhood, Airbnb Sydney", fontsize=20, pad=60)

plt.savefig("airbnb_price_suburb.png", bbox_inches='tight', pad_inches=0.1)
# Plot the data, Avg. Rating Choropleth



gplt.choropleth(syd_geo, hue=syd_geo['review_scores_rating'], projection=gcrs.PlateCarree(),

                cmap='Blues', linewidth=0.5, edgecolor='white', k=None, legend=True, figsize=(10, 10))

plt.title("Avg. Rating per Neighbourhood, Airbnb Sydney", fontsize=20, pad=60)

plt.savefig("airbnb_ratings_suburb.png", bbox_inches='tight', pad_inches=0.1)