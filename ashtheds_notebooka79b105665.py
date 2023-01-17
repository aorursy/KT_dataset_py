import pandas as pd

import numpy as np

import folium

from geopy.geocoders import Nominatim

import json

import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

import matplotlib.pyplot as plt

from geopy.extra.rate_limiter import RateLimiter
tweets_df=pd.read_csv('../input/covid19-tweets/covid19_tweets.csv',header=0)
tweets_df.head()
tweets_df.info()
tweets_df.describe()
tweets_df.dropna(inplace=True,subset=['user_location','hashtags'])

tweets_df.info()
tweets_df = tweets_df.replace('[^a-zA-Z0-9 ]', '', regex=True)

tweets_df.head()
tweets_df.reset_index(inplace=True,drop=True)

tweets_df.head()
"""geolocator = Nominatim(user_agent="ny_explorer")

geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

longitude=[]

latitude=[]



#address=['chennai','dallas','hsjhgdgh','London, UK']



for i in tweets_df['user_location'].astype(str)[0:10]:

    location = geolocator.geocode(i)

    if location:

        latitude.append(location.latitude)

        longitude.append(location.longitude)

    else:

        latitude.append(np.NaN)

        longitude.append(np.NaN)

    

    

print(len(latitude))    """
tweets_df.isnull().any()
tweets_df['date']=pd.to_datetime(tweets_df['date'])
top_july = tweets_df['user_location'][pd.DatetimeIndex(tweets_df['date']).month == 7].value_counts()
top_august = tweets_df['user_location'][pd.DatetimeIndex(tweets_df['date']).month == 8].value_counts()

top_all_the_time = (top_august + top_july).sort_values(ascending = False)
fig, ax = plt.subplots(figsize = (13,5))

plt.xlabel("Location", fontsize = 12)

plt.ylabel("NO. Tweets", fontsize = 12)

top_july[0:10].plot(kind='bar', title = "Top 10 Countries Posting about Covid-19 in July" )
fig, ax=plt.subplots(figsize=(13,5))

plt.xlabel("Top Locations")

plt.ylabel("Tweet Count")

plt.title("Top 10 Countries Posting about Covid-19 in August")

top_august[0:10].plot(kind='bar')
a=tweets_df['source'].value_counts()
a[0:5].plot(kind='bar')
tags=tweets_df['hashtags'].value_counts()
tags[0:10].plot(kind='bar')
user_status=tweets_df['user_verified'].value_counts()
user_status.plot(kind='bar')

plt.xlabel("Account verification status")

plt.ylabel("Number of tweets")

plt.title("Verified account tweets vs Unverified account tweets")

plt.xticks([False,True],['Unverified','Verified'])
pwd