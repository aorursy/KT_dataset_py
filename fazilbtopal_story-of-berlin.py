import pandas as pd

import numpy as np



import os

print(os.listdir("../input/berlin-airbnb-data/"))
listings = pd.read_csv("../input/berlin-airbnb-data/listings.csv", index_col= "id")

listings_summary = pd.read_csv("../input/berlin-airbnb-data/listings_summary.csv", index_col= "id")

calendar_summary = pd.read_csv("../input/berlin-airbnb-data/calendar_summary.csv", parse_dates=['date'], index_col='listing_id')

reviews = pd.read_csv("../input/berlin-airbnb-data/reviews.csv", parse_dates=['date'], index_col='listing_id')

reviews_summary = pd.read_csv("../input/berlin-airbnb-data/reviews_summary.csv", parse_dates=['date'], index_col='id')
print(listings.shape)
listings.head()
listings_summary.info()
listings_summary.columns
target_columns = ["property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", 

                  "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin",

                  "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time",

                  "host_response_rate", "street", "weekly_price", "monthly_price", "market"]

listings = pd.merge(listings, listings_summary[target_columns], on='id', how='left')

listings.info()
listings['host_response_rate'] = pd.to_numeric(listings.host_response_rate.str.strip('%'))

listings.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

sns.set_palette(palette='deep')

import folium

from folium.plugins import FastMarkerCluster
freq = listings['neighbourhood_group'].value_counts().sort_values(ascending=True)

freq.plot.barh(figsize=(10, 8), width=1)

plt.title("Number of Listings by Neighbourhood Group")

plt.xlabel('Number of Listings')

plt.show()
lat = listings['latitude'].tolist()

lon = listings['longitude'].tolist()

locations = list(zip(lat, lon))



map1 = folium.Map(location=[52.5200, 13.4050], zoom_start=12)

FastMarkerCluster(locations).add_to(map1)

map1
listings.property_type.unique()
listings.room_type.unique()
freq = listings['room_type'].value_counts().sort_values(ascending=True)

freq.plot.barh(figsize=(10, 5), width=1)

plt.show()
freq = listings['property_type'].value_counts().sort_values(ascending=True)

freq = freq[freq > 20]  # Eliminate types less than 20 counts.

freq.plot.barh(figsize=(15, 8), width=1)

plt.xscale('log')

plt.show()
freq = listings['accommodates'].value_counts().sort_index()

freq.plot.bar(figsize=(12, 8), width=1, rot=0)

plt.title("Number of People")

plt.ylabel('Number of Listings')

plt.xlabel('Accommodates')

plt.show()
freq = listings[listings['accommodates']==2]

freq = freq.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=True)

freq.plot.barh(figsize=(12, 8), width=1)

plt.title("Average Daily Price for 2 People")

plt.xlabel('Average Daily Price (Dollar)')

plt.ylabel("Neighbourhodd")

plt.show()
listings.host_is_superhost = listings.host_is_superhost.replace({"t": "True", "f": "False"})

freq=listings['host_is_superhost'].value_counts()

freq.plot.bar(figsize=(10, 8), width=1, rot=0)

plt.title("Number of Listings with Superhost")

plt.ylabel('Number of Listings')

plt.show()
listings10 = listings[listings['number_of_reviews']>=10]

fig = plt.figure(figsize=(16,10))



ax = fig.add_subplot(121)

freq = listings10['host_response_rate'].dropna()

freq.plot.hist('host_response_rate', ax=ax)

plt.title("Response Rate")

plt.ylabel("number of listings")

plt.xlabel("Percent")



ax = fig.add_subplot(122)

freq = listings10['host_response_time'].value_counts()

freq.plot.bar(width=1, rot=45, ax=ax)

plt.title("Response Time")

plt.ylabel("Number of Listings")



plt.tight_layout()

plt.show()
calendar_summary.head()
calendar_summary.price = calendar_summary.price.str.replace(",","")

calendar_summary.price = pd.to_numeric(calendar_summary.price.str.strip('$'))

calendar_summary = calendar_summary[calendar_summary.date < '2019-12-30']



listings.index.name = "listing_id"

calendar = pd.merge(calendar_summary, listings[['accommodates']], on="listing_id", how="left")

calendar.sample(10)
sum_available = calendar[calendar.available == "t"].groupby(['date']).size().to_frame(name='available').reset_index()

sum_available = sum_available.set_index('date')



sum_available.plot(kind='line', y='available', figsize=(12, 8))

plt.title('Number of Listings Available by Date')

plt.xlabel('Date')

plt.ylabel('Number of Listings Available')