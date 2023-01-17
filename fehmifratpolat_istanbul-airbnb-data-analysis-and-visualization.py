import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import folium
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/airbnb-istanbul-dataset/AirbnbIstanbul.csv")
df.head()
df.info()
df.dtypes
df.isnull().sum()
df.drop(columns="neighbourhood_group", inplace = True)
df.drop(columns="last_review", inplace = True)
df.head()
df["reviews_per_month"].fillna(0, inplace=True)
df.head()
df["neighbourhood"].unique()
df["neighbourhood"].value_counts()
df["room_type"].unique()
df["room_type"].value_counts()
df["host_id"].value_counts().head(10)
chart1 = df["host_id"].value_counts().head(10).plot(kind="bar")
chart1.set_title(" Hosts with Most Listings")
chart1.set_xlabel("Host ID")
chart1.set_ylabel("Number of Listings")
chart2 = df["neighbourhood"].value_counts().head(10).plot(kind="bar")
chart2.set_title(" Neighbourhoods with Most Listings")
chart2.set_xlabel("Neighbourhoods")
chart2.set_ylabel("Number of Listings")
neighbourhood_price = df.groupby("neighbourhood")["price"].agg(['mean'])
neighbourhood_price.sort_values(by='mean', ascending=False)
chart3 = neighbourhood_price.sort_values(by='mean', ascending=False).head(10).plot(kind = "bar")
chart3.set_ylabel('Price TRY')
chart3.set_xlabel('Neighbourhoods')
chart3.set_title("Average Price of Neighbourhoods")
most_expensive_prices = df[df.price>5000].sort_values(by="price", ascending=False).head(10)
most_expensive_prices
city_pos = [df.latitude.mean(),df.longitude.mean()]
istanbul_map = folium.Map(location=city_pos, zoom_start=10)
folium.Marker(
    location=[41.03740, 28.79435],
    popup='3 Rooms 1 Living Room - Grand Holiday Istanbul',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.03841, 28.79471],
    popup='3 Rooms 1 Living Room Dublex - Grand Holiday Istanbul',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.00445, 28.97907],
    popup='Elegance Single Room - Avicenna Hotel',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.02681, 28.62680],
    popup='Gunluk kiralik daire',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.00850, 28.96649],
    popup='Istanbul town history place ',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.03015, 28.98064],
    popup='CoZy room in Beyoğlu/cihangir',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.05465, 28.98111],
    popup='İstanbul un kalbi sisli. Center of istanbul sisli',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.03383, 28.97151],
    popup='Private room in Beyoğlu(nice view)',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[40.99484, 29.02976],
    popup='hmgv',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)
folium.Marker(
    location=[41.05709, 28.98525],
    popup='Room in the center BOMONTI',
    icon=folium.Icon(icon='cloud')).add_to(istanbul_map)

istanbul_map
ave_price=df[df.price < 1500]
chart4=ave_price.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,6))
chart4.set_title('Price Map of Istanbul')
chart4.set_xlabel('longitude')
chart4.legend()
most_reviewed = df.nlargest(10, "number_of_reviews")
most_reviewed
price_avrg=most_reviewed.price.mean()
print('Average price per night: {}'.format(price_avrg))
df[df["host_id"]==21907588].head(5)