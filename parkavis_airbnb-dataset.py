#importing required libraries
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import math
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
#importing the dataset
airbnb = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
#exploring the dataset
airbnb.head(10)
airbnb.describe()
airbnb.info()
airbnb.shape
airbnb.isnull().sum()
#preprocessing the dataset
airbnb['reviews_per_month'].fillna(value = 0 , inplace = True)
airbnb.drop(['id' , 'host_id' , 'host_name' ,'last_review'] , axis = 1 , inplace = True)
airbnb.head()
sns.catplot(x="neighbourhood_group", kind = "count", data = airbnb)
plt.show()
sns.catplot(x="neighbourhood_group", kind = "count", data = airbnb)
plt.show()
neighbourhood_top10 = airbnb["neighbourhood"].value_counts().head(10)
df_neighbourhood_top10 = pd.DataFrame(neighbourhood_top10)
df_neighbourhood_top10 = df_neighbourhood_top10.reset_index()
f, ax = plt.subplots(figsize = (15,5))
sns.barplot(x ="index", y = "neighbourhood" ,data = df_neighbourhood_top10)
plt.show()
airbnb_price = airbnb.groupby(["room_type"])["price"].median()
df_airbnb_price = pd.DataFrame(airbnb_price)
df_airbnb_price = df_airbnb_price.reset_index()

sns.catplot(x="room_type", y="price",kind = "bar", palette = "Accent",  data = df_airbnb_price)
plt.title("Room_type price by it's median")
plt.show()
airbnb_reviews = airbnb.groupby(["neighbourhood_group"])["number_of_reviews"].sum()
df_airbnb_reviews = pd.DataFrame(airbnb_reviews)
df_airbnb_reviews = df_airbnb_reviews.reset_index()

sns.scatterplot(x="neighbourhood_group", y="number_of_reviews", data = df_airbnb_reviews)
plt.title("Total Reviews by neighbourhood_group")
plt.show()
airbnb_night = airbnb.groupby(["neighbourhood_group"])["minimum_nights"].mean().round(2)
df_airbnb_night = pd.DataFrame(airbnb_night)
df_airbnb_night = df_airbnb_night.reset_index()
sns.catplot(x="minimum_nights", y = "neighbourhood_group",kind="bar",data = df_airbnb_night)
plt.title("Minimum_nights mean by neighbourhood_group")
plt.show()
airbnb_proportion = airbnb.groupby(["neighbourhood_group"])["room_type"].value_counts()
df_airbnb_proportion = pd.DataFrame(airbnb_proportion)
df_airbnb_proportion.rename(columns={"room_type":"Total of values"}, inplace = True)


airbnb_count = airbnb.groupby(["neighbourhood_group"])["room_type"].count()
df_airbnb_count = pd.DataFrame(airbnb_count)


df_airbnb_proportion["Total"] = 0

df_airbnb_proportion.loc["Bronx"]["Total"]= df_airbnb_count.room_type.loc["Bronx"]
df_airbnb_proportion.loc["Brooklyn"]["Total"]= df_airbnb_count.room_type.loc["Brooklyn"]
df_airbnb_proportion.loc["Manhattan"]["Total"]= df_airbnb_count.room_type.loc["Manhattan"]
df_airbnb_proportion.loc["Queens"]["Total"]= df_airbnb_count.room_type.loc["Queens"]
df_airbnb_proportion.loc["Staten Island"]["Total"]= df_airbnb_count.room_type.loc["Staten Island"]

df_airbnb_proportion = df_airbnb_proportion.reset_index()

df_airbnb_proportion["Proportion"] = (df_airbnb_proportion["Total of values"]/df_airbnb_proportion["Total"]).round(2)

sns.catplot(x="neighbourhood_group",
            y = "Proportion",
            kind = "bar",
            hue = "room_type",
            data = df_airbnb_proportion)
plt.title("Room_type proportion for each neighbourhood_group")
plt.show()
sns.relplot(x="latitude", y="longitude", palette = "Set2", hue = "neighbourhood_group", data = airbnb)
plt.show()
corr = airbnb.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.figure(figsize = (15, 15))
plt.style.use('seaborn-white')
plt.subplot(221)
sns.scatterplot(x="latitude", y="longitude",hue="neighbourhood_group", data=airbnb)
plt.subplot(222)
sns.scatterplot(x="latitude", y="longitude",hue="room_type", data=airbnb)
plt.subplot(223)
sns.scatterplot(x="latitude", y="longitude",hue="price", data=airbnb)
plt.subplot(224)
sns.scatterplot(x="latitude", y="longitude",hue="availability_365", data=airbnb)
plt.show()
geomap = folium.Map(location=[40.7128,-74.0060], tiles='cartodbpositron', zoom_start=12)
# Adding a heatmap to the base map
HeatMap(data=airbnb[['latitude', 'longitude']], radius=10).add_to(geomap)
geomap