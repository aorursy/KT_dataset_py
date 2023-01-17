import pandas as pd

import numpy as np

import matplotlib as mlt

import matplotlib.pyplot as plt

import seaborn as sb

import folium

import os

from datetime import datetime
# The purpose of this project is to do a descriptive analysis of the Airbnb open source data
# Data source link



# http://insideairbnb.com/get-the-data.html



# Datasets used:



# listings_summary.csv



# calendar.csv



# neighbourhoods.csv



# reviews.csv
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing datasets



listings = pd.read_csv('/kaggle/input/airbnb-amsterdam-data/listing_summary.csv')
listings.head()
# Shape of the dataset



listings.shape
# Listings by room type



by_room_type = listings.groupby(by = 'room_type')[listings.columns[2]].count().sort_values(ascending=False)



by_room_type
type(by_room_type)
by_room_type.index
sb.barplot(by_room_type.values, by_room_type.index,);
# There are more number of Entire home/apt listed as compared to others
# But how much does every room type cost on an average ?
# Using boxplot to describe this



sb.boxplot(x="price", y="room_type", data=listings,

            whis=[0, 100], palette="vlag");
# There is a huge variability in the prices, so averages would be skewed. Lets look at median prices by room type
median_vals = listings.groupby(by = 'room_type')[listings.columns[9]].median().sort_values(ascending = False)



median_vals
# So hotel rooms are most expensive in amsterdam
# Lets check out busiest properties in Amsterdam(The ones having lowest availability
max_availability = listings[['name','availability_365']].sort_values(by = 'availability_365', ascending = False)
max_availability.head(20)
min_availability = listings[['name','availability_365']].sort_values(by = 'availability_365')
min_availability.head(10)
# Not a useful exercise, looks like a lot of properties have round the year availability, while a lot others have none
# How about looking at histogram of availabiltiy
sb.distplot(listings[['availability_365']], bins = 365, kde = False);
# Most of the properties hace no availability
# Prices and reviews vary by neighbourhood, but lets find out which neighbourhood are highly rated and expensive
prices_reviews = listings.groupby(by = 'neighbourhood', as_index=False).agg({'price':'mean','reviews_per_month':'mean', 'id':'count'}).sort_values(by='neighbourhood', ascending = False)

prices_reviews.head(10)
# Building a scatterplot

plot = sb.relplot(x="reviews_per_month", y="price", size = "id", hue = "neighbourhood", data = prices_reviews)
# Seems that there are few neighbourhoods with high # of reviews and low price but there are none with less # of reviews with high price
# Price and minimum number of nights are generally having an inverse relationship. Lets see if it is true for our airbnb listings
price_nights = listings.groupby(by='room_type').agg({'price':'mean', 'minimum_nights':'mean'}).sort_values(by = 'price', ascending = False)

price_nights
# This makes sense as entire apts require more minimum nights stay and are hence, also less expensive

# Drawing this out on a graph
price_nights.index
# ax1 = sb.barplot(price_nights.index, price_nights.price,)
# ax2 = ax1.twinx()
# ax3 = ax2.plot(price_nights.index, price_nights.minimum_nights,)
# plt.show()
# Also, lets look at the histogram of prices and derive some insights



sb.distplot(listings[['price']], bins = 900, kde = False);
# Clearly there are a few outliers here and it seems that 1000 probably is a good cut-off for any predictive analysis
# Lets explore further
listings['price'].describe()
pbins = pd.qcut(listings['price'], q=[0,0.4,0.6,0.9,0.98,1])

pbins.value_counts()
listings['price_bins']=pbins
price_bins = listings.groupby(by = 'price_bins').agg({'id':'count'}).sort_values(by = 'price_bins', ascending=False)

price_bins
sb.barplot(price_bins.index, price_bins.id,);
# It seems really most of the listings are within 0-$480 price and very few are above it. Its safe to consider those as outliers.
# Lets also bin availability and see what we get



abins = pd.cut(listings['availability_365'], bins = 4)

abins.value_counts()
listings['availability_bins']=abins
availability_bins=listings.groupby(by='availability_bins').agg({'id':'count'}).sort_values(by = 'availability_bins', ascending = False)

availability_bins
sb.barplot(availability_bins.index, availability_bins.id,);
# Clearly there are a lot of listings with low availability. We should also check how many of them are zero for data sanity
listings['id'][listings['availability_365']==0].count()*100/listings['id'].count()
# 60 percent of the listings have 0 availability throughout the year. This is a little fishy and in real world scenario, we would have checked this with the data collection team
# It might also be worthwhile looking at number of listings by neighbourhood
neigh = listings.groupby(by = 'neighbourhood').agg({'id':'count'}).sort_values(by = 'neighbourhood', ascending = False)

neigh
# plotting this on a graph



# Top 5 neighbourhoods

neigh_head = neigh.head(5)



neigh_head_plot = sb.barplot(neigh_head.index, neigh_head.id,)

neigh_head_plot.set_xticklabels(neigh_head_plot.get_xticklabels(),rotation=90);
# Bottom 5 neighbourhoods

neigh_tail = neigh.tail(5)



neigh_tail_plot = sb.barplot(neigh_tail.index, neigh_tail.id,)

neigh_tail_plot.set_xticklabels(neigh_tail_plot.get_xticklabels(),rotation=90);
listings['last_review']=pd.to_datetime(listings['last_review'])

listings['Mon_Year'] = listings['last_review'].dt.strftime('%Y-%m')
listings.Mon_Year.head(10)
last_rev = listings.groupby(by = 'Mon_Year').agg({'Mon_Year':'count'}).sort_index().tail(36)

last_rev.tail(10)
plt.figure(figsize=(15,8))

last_rev_plot = sb.lineplot(last_rev.index, last_rev.Mon_Year,)

last_rev_plot.set_xticklabels(last_rev_plot.get_xticklabels(),rotation=90);
# reviews are increasing per month over time but probably it is a better idea to look at this by plotting different lines for each year to determine seasonality
# For that we will need to create month and year in listings itself



listings['month'] = listings['last_review'].dt.strftime('%b')

listings.month.head(10)
listings['Year'] = listings['last_review'].dt.strftime('%Y')

listings.Year.head(10)
months_in_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
review_count = listings.groupby(by=["Year", "month"]).size().reset_index()

review_count.tail(10)
review_count.columns
review_count.columns = ['Year','month','reviews']
reviews_pivoted = review_count[review_count['Year']>'2014'].pivot(index='month', columns='Year', values='reviews').reindex(months_in_order)

reviews_pivoted2 = reviews_pivoted.iloc[:,:-2]

reviews_pivoted2
reviews_pivoted2.plot(kind='bar', figsize=(15,8), color=['blue', 'green', 'orange', 'purple', 'black'])

plt.title("Count of reviews by month and year", y=1.013, fontsize=22)

plt.xlabel("Month", labelpad=16)

plt.ylabel("Count of reviews", labelpad=16);  
# Seems like fall is the visiting season and there are a lot of reviews Jun- Nov which is our proxy for booking dates
# Lets confirm this observation by a line chart



reviews_pivoted2.plot(kind='line', figsize=(15,8), color=['blue', 'green', 'orange', 'purple', 'black'])

plt.title("Count of reviews by month and year", y=1.013, fontsize=22)

plt.xlabel("Month", labelpad=16)

plt.ylabel("Count of reviews", labelpad=16);  
# So our observation is true and there is definitely a seasonality to this. Higher travel and hence, higher reviews during fall season.
# But do the reviews have to do anything with which neighbourhood it is ? Lets have a lookb
rev_nb = listings.groupby(by='neighbourhood').agg({'id':'count', 'number_of_reviews':'mean'}).sort_values(by='number_of_reviews', ascending=False).reset_index()

rev_nb
# So it seems there are neighbourhoods with less listings and more number of reviews. What's going on there ? 

# Lets see if there is a relation to average price in these neighbourhoods
rev_nb_p = listings.groupby(by='neighbourhood').agg({'id':'count', 'number_of_reviews':'mean', 'price':'mean'}).sort_values(by='number_of_reviews', ascending=False).reset_index()

rev_nb_p
# Not much of a conclusion here. Lets check the correlation to be sure
corrmatrix = rev_nb_p.corr()

corrmatrix
# There is mild correlation between listings and average price, not much correlation of number of reviews against any of the other metrics though

# Lets visualize this
sb.heatmap(corrmatrix, annot = True);
# Hence, we can safely conclude that there isn't much of a correlation here
# Lets see the listings data on an interactive map
folium_map = folium.Map(location=[52.367, 4.894],

                        zoom_start=13)

marker = folium.CircleMarker(location=[40.738, -73.98])

marker.add_to(folium_map)



for index, row in listings.iterrows():

    radius = 1 # row["number_of_reviews"]/1000

    if row["price"]>0 and row["price"]<=120:

        color="#64FF33" # green

    elif row["price"]>120 and row["price"]<=150:

        color="#33FFEC" #blue

    elif row["price"]>150 and row["price"]<=260:

        color="#FFFF33" # yellow

    else:

        color="#FF3933" # red

    

    folium.CircleMarker(location=(row["latitude"],

                                  row["longitude"]),

                        radius=radius,

                        color=color,

                        fill=True).add_to(folium_map)

folium_map



# (-0.001, 120.0]    8290

# (150.0, 260.0]     5082

# (120.0, 150.0]     4066

# (260.0, 480.0]     1539

# (480.0, 9000.0]     385
# From the map, it is clear that price segregation is less in amsterdam 

# but still more expensive properties are located in the heart of the city whereas less expensive ones are located in the outskirts
# That concludes my analysis on the Airbnb Dataset of Amsterdam. This will be forked into a predictive analysis soon.

# Feel free to share any feedback - goor or constructive