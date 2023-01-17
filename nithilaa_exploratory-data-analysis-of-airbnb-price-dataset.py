# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
dataset.head(10)
dataset.describe()
dataset.info()
dataset.shape
dataset.isnull().sum()
dataset['reviews_per_month'].fillna(value = 0 , inplace = True)

dataset.drop(['id' , 'host_id' , 'host_name' ,'last_review'] , axis = 1 , inplace = True)

dataset.head()
sns.catplot(x="neighbourhood_group", kind = "count", palette = "Set2", data = dataset)

plt.show()
neighbourhood_top10 = dataset["neighbourhood"].value_counts().head(10)

df_neighbourhood_top10 = pd.DataFrame(neighbourhood_top10)

df_neighbourhood_top10 = df_neighbourhood_top10.reset_index()

f, ax = plt.subplots(figsize = (15,5))

sns.barplot(x ="index", y = "neighbourhood", palette = "Set2", data = df_neighbourhood_top10)

plt.show()
dataset_price = dataset.groupby(["room_type"])["price"].median()

df_dataset_price = pd.DataFrame(dataset_price)

df_dataset_price = df_dataset_price.reset_index()



sns.catplot(x="room_type", y="price", kind = "bar", palette = "Accent",  data = df_dataset_price)

plt.title("Room_type price by it's median")

plt.show()
dataset_reviews = dataset.groupby(["neighbourhood_group"])["number_of_reviews"].sum()

df_dataset_reviews = pd.DataFrame(dataset_reviews)

df_dataset_reviews = df_dataset_reviews.reset_index()



sns.barplot(x="neighbourhood_group", y="number_of_reviews", data = df_dataset_reviews)

plt.title("Total Reviews by neighbourhood_group")

plt.show()
dataset_night = dataset.groupby(["neighbourhood_group"])["minimum_nights"].mean().round(2)

df_dataset_night = pd.DataFrame(dataset_night)

df_dataset_night = df_dataset_night.reset_index()



sns.catplot(x="minimum_nights", y = "neighbourhood_group", kind = "bar", data = df_dataset_night)

plt.title("Minimum_nights mean by neighbourhood_group")

plt.show()
dataset_proportion = dataset.groupby(["neighbourhood_group"])["room_type"].value_counts()

df_dataset_proportion = pd.DataFrame(dataset_proportion)

df_dataset_proportion.rename(columns={"room_type":"Total of values"}, inplace = True)





dataset_count = dataset.groupby(["neighbourhood_group"])["room_type"].count()

df_dataset_count = pd.DataFrame(dataset_count)





df_dataset_proportion["Total"] = 0



df_dataset_proportion.loc["Bronx"]["Total"]= df_dataset_count.room_type.loc["Bronx"]

df_dataset_proportion.loc["Brooklyn"]["Total"]= df_dataset_count.room_type.loc["Brooklyn"]

df_dataset_proportion.loc["Manhattan"]["Total"]= df_dataset_count.room_type.loc["Manhattan"]

df_dataset_proportion.loc["Queens"]["Total"]= df_dataset_count.room_type.loc["Queens"]

df_dataset_proportion.loc["Staten Island"]["Total"]= df_dataset_count.room_type.loc["Staten Island"]



df_dataset_proportion = df_dataset_proportion.reset_index()



df_dataset_proportion["Proportion"] = (df_dataset_proportion["Total of values"]/df_dataset_proportion["Total"]).round(2)



sns.catplot(x="neighbourhood_group",

            y = "Proportion",

            kind = "bar",

            hue = "room_type",

            data = df_dataset_proportion)

plt.title("Room_type proportion for each neighbourhood_group")

plt.show()
sns.relplot(x="latitude", y="longitude", palette = "Set2", hue = "neighbourhood_group", data = dataset)

plt.show()
corr = dataset.corr()

sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.figure(figsize = (15, 15))

plt.style.use('seaborn-white')

plt.subplot(221)

sns.scatterplot(x="latitude", y="longitude",hue="neighbourhood_group", data=dataset)

plt.subplot(222)

sns.scatterplot(x="latitude", y="longitude",hue="room_type", data=dataset)

plt.subplot(223)

sns.scatterplot(x="latitude", y="longitude",hue="price", data=dataset)

plt.subplot(224)

sns.scatterplot(x="latitude", y="longitude",hue="availability_365", data=dataset)

plt.show()
import pandas as pd

import geopandas as gpd

import math

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

m_1 = folium.Map(location=[40.7128,-74.0060], tiles='cartodbpositron', zoom_start=12)



# Adding a heatmap to the base map

HeatMap(data=dataset[['latitude', 'longitude']], radius=10).add_to(m_1)



# Displaying the map

m_1