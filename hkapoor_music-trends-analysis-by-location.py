import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data_df = pd.read_csv("/kaggle/input/spotify-top-songs-by-country-may-2020/SpotifyTopSongsByCountry - May 2020.csv")
data_df.shape
data_df.head(5)
unique_countries = data_df["Country"].unique()

unique_countries = unique_countries[unique_countries != "Global"]

unique_countries
from geopy.geocoders import Nominatim



latitude = []

longitude = []

geolocator = Nominatim(user_agent="my-app")



for i in unique_countries:

    location = geolocator.geocode(i)

    latitude.append(location.latitude)

    longitude.append(location.longitude)
# reference - https://python-graph-gallery.com/313-bubble-map-with-folium/



import folium

 

data = pd.DataFrame({

   'lat':latitude,

   'lon':longitude,

   'name':unique_countries,

    'value':[20.0]*62

})

 

m = folium.Map(location=[20,0], tiles="OpenStreetMap", zoom_start=2)

 

    

for i in range(0,len(data)):

    folium.Circle(

      location=[data.iloc[i]['lat'], data.iloc[i]['lon']],

      popup=data.iloc[i]['name'],

      radius=data.iloc[i]['value']*10000,

      color='#1db954',

      fill=True,

      fill_color='#1db954'

   ).add_to(m)

    

m
data_df["Continent"].unique()
continent_value_counts = (data_df["Continent"].value_counts()/50).astype("int32")

continent_value_counts = continent_value_counts.drop("Global")
continent_value_counts
height = continent_value_counts.values

bars = continent_value_counts.index

y_pos = range(0,12,2)



fig = plt.figure(figsize=[13,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.bar(y_pos, height, color="#1db954", width=1.2)

 

plt.xticks(y_pos, bars, color="#424242")

plt.yticks(color="#424242")

for i, v in enumerate(height):

    ax.text((i)*2 - 0.1, v+0.5, str(v), color='#424242')

plt.title("Number of countries in each continent", y=-0.15)



plt.show()
top10_tracks = data_df["Title"].value_counts()[:10].sort_values(ascending=True)
height = top10_tracks.values

bars = top10_tracks.index

y_pos = np.arange(len(bars))



fig = plt.figure(figsize=[11,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible("#424242")

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.barh(y_pos, height, color="#1db954", height=0.8)

 

plt.xticks(color="#424242")

plt.yticks(y_pos, bars, color="#424242")

plt.xlabel("Number of occurances in charts")



for i, v in enumerate(height):

    ax.text(v+1, i, str(v), color='#424242')

plt.title("Top 10 Tracks")





plt.show()
Artists = []

for i in data_df["Artists"]:

    a = i.split(", ")

    Artists = Artists + a
len(Artists)
top10_artists = pd.Series(Artists).value_counts()[:10].sort_values(ascending=True)
height = top10_artists.values

bars = top10_artists.index

y_pos = np.arange(len(bars))



fig = plt.figure(figsize=[13,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible("#424242")

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.barh(y_pos, height, color="#1db954", height=0.8)

 

plt.xticks(color="#424242")

plt.yticks(y_pos, bars, color="#424242")

plt.xlabel("Number of artist occurances in charts")



for i, v in enumerate(height):

    ax.text(v+1, i, str(v), color='#424242')

plt.title("Top 10 Artists")





plt.show()
top10_albums = data_df["Album"].value_counts()[:10].sort_values(ascending=True)
height = top10_albums.values

bars = top10_albums.index

y_pos = np.arange(len(bars))



fig = plt.figure(figsize=[11,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible("#424242")

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.barh(y_pos, height, color="#1db954", height=0.8)

 

plt.xticks(color="#424242")

plt.yticks(y_pos, bars, color="#424242")

plt.xlabel("Number of album occurances in charts")



for i, v in enumerate(height):

    ax.text(v+1, i, str(v), color='#424242')

plt.title("Top 10 Albums")



plt.show()
seconds = []

for i in data_df["Duration"]:

    val = i.split(":")

    secs = int(val[0])*60 + int(val[1])

    seconds.append(secs)
fig = plt.figure(figsize=[13,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



sns.distplot(seconds, hist=True, kde=True, bins=int(180/5), color = '#1db954',  hist_kws={'edgecolor':'#1db954'},

             kde_kws={'linewidth': 4})

 

plt.xticks(color="#424242")

plt.yticks(color="#424242")

plt.xlabel("Length of song(in seconds)")

plt.title("PDF of Song duration")



plt.show()
data_df["duration_in_s"] = seconds

duration_by_c = data_df.groupby("Country").mean()["duration_in_s"]
bottom5_duration = duration_by_c.sort_values()[:5]

bottom5_duration
top5_duration = duration_by_c.sort_values(ascending=False)[:5]

top5_duration
filtered_data1 = data_df[(data_df["Country"].isin(list(top5_duration.index))) + (data_df["Country"].isin(list(bottom5_duration.index)))]

fig = plt.figure(figsize=[13,10], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



sns.boxplot( x = filtered_data1["Country"], y = filtered_data1["duration_in_s"], color = '#1db954')

 

plt.xticks(color="#424242")

plt.yticks(color="#424242")

plt.xlabel("Length of song(in seconds)")

plt.title("Box plot of average song duration in top and bottom 5 countries")



plt.show()
filtered_data2 = data_df[data_df["Continent"] != "Global"]
fig = plt.figure(figsize=[13,10], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



sns.boxplot( x = filtered_data2["Continent"], y = filtered_data2["duration_in_s"], color = '#1db954')

 

plt.xticks(color="#424242")

plt.yticks(color="#424242")

plt.xlabel("Length of songs (in seconds)")

plt.title("Box plot of average song duration in top and bottom 5 countries")



plt.show()
exp_data = data_df["Explicit"].value_counts().sort_values()

exp_data
height = exp_data.values

bars = ["Explicit", "Not explicit"]

y_pos = range(0,4,2)



fig = plt.figure(figsize=[8,5], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.bar(y_pos, height, color="#1db954", width=0.8)

 

plt.xticks(y_pos, bars, color="#424242")

plt.yticks(color="#424242")

for i, v in enumerate(height):

    ax.text((i)*2 - 0.1, v+30, str(v), color='#424242')

plt.title("Explicit songs", y=-0.15)



plt.show()
exp_data_cont = data_df.groupby("Continent").mean()["Explicit"]

exp_data_cont
height = exp_data_cont.values

bars = exp_data_cont.index

y_pos = range(0,14,2)



fig = plt.figure(figsize=[13,7], frameon=False)

ax = fig.gca()

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#424242")

ax.spines["bottom"].set_color("#424242")



plt.bar(y_pos, height, color="#1db954", width=0.8)

plt.bar(y_pos, 1 - height, bottom = height,color="#3CDC75", width=0.8)

 

plt.xticks(y_pos, bars, color="#424242")

plt.yticks(color="#424242")

'''

for i, v in enumerate(height):

    ax.text((i)*2 - 0.2, v+0.01, str(np.round(v,2)), color='#424242')

'''

plt.title("Explicit songs percentage by continents", y=-0.15)



plt.show()