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

tweets_df.replace('',np.NaN)
tweets_df.info()
tweets_df.dropna(inplace=True)
tweets_df.info()
tweets_df['user_name'].value_counts()
top_users=tweets_df.groupby('user_name')['user_location'].count().reset_index()
top_users.columns=['user_name','count']
top_users.sort_values('count',ascending=False,inplace=True)
top_users[0:10].plot(kind='bar',x='user_name',y='count')
plt.xlabel('Users')
plt.ylabel('Tweets')
plt.title('Top 10 tweeters')
plt.show()
locations=tweets_df['user_location'].replace('[^a-zA-Z0-9 ]', '', regex=True)
tweets_df['user_location']=locations
tweets_df.reset_index(inplace=True,drop=True)
tweets_df.head()
tweets_df.dropna(inplace=True)
tweets_df.info()


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
tweets_df2=tweets_df.drop_duplicates(subset='user_name')
tweets_df2=tweets_df2[['user_name','user_location']]
tweets_df2.set_index('user_name',inplace=True)
tweets_df2.head()
to_plot_list=top_users['user_name']
to_plot_list=to_plot_list[0:100]
#to_plot_list.shape
to_plot_map=tweets_df2.loc[to_plot_list]
#to_plot_map.columns
#to_plot_map.shape
geolocator = Nominatim(user_agent="ny_explorer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
longitude=[]
latitude=[]

#address=['chennai','dallas','hsjhgdgh','London, UK']

for i in to_plot_map['user_location'].astype(str):
    location = geolocator.geocode(i)
    if location:
        latitude.append(location.latitude)
        longitude.append(location.longitude)
    else:
        latitude.append(np.NaN)
        longitude.append(np.NaN)
    
    
print(len(latitude)) 

world_map = folium.Map(zoom_start=14)

latitude=list(latitude)
longitude=list(longitude)

cleanedlatitude = [x for x in latitude if str(x) != 'nan']

cleanedlongitude = [x for x in longitude if str(x) != 'nan']
incident_tweets=folium.map.FeatureGroup()

for i, j in zip(cleanedlatitude, cleanedlongitude):
    incident_tweets.add_child(
        folium.CircleMarker(
            [i, j],
            radius=5, # define how big you want the circle markers to be
            color='red',
            fill=True,
            fill_color='green',
            fill_opacity=0.5,
            
        
        )
    )
    
world_map.add_child(incident_tweets)