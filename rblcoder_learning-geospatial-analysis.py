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
import geopandas as gpd

df = gpd.read_file('/kaggle/input/munich-airbnb-listings-data-may-24-2020/neighbourhoods.geojson')

df.head()
df.info()
df.geometry.area
df['neighbourhood'].nunique()
df.plot( color='none', edgecolor='gainsboro', zorder=3)
# https://residentmario.github.io/geoplot/user_guide/Working_with_Geospatial_Data.html

df_list = pd.read_csv('/kaggle/input/munich-airbnb-listings-data-may-24-2020/listings.csv')

#df_list = gpd.read_file('/kaggle/input/munich-airbnb-listings-data-may-24-2020/listings.csv')
(df_list.nunique()).sort_values()[:30]
df_list.info()
(df_list.nunique()).sort_values()[-30:]
from shapely.geometry import Point



points = df_list.apply(

    lambda srs: Point(float(srs['longitude']), float(srs['latitude'])),

    axis='columns'

)

points
df_list_geocoded = gpd.GeoDataFrame(df_list, geometry=points)

df_list_geocoded
df_list_geocoded.plot()
ax = df.plot(figsize=(10,10), color='whitesmoke', linestyle=':', edgecolor='black')

df_list_geocoded.plot(markersize=1, ax=ax)
# https://www.earthdatascience.org/courses/scientists-guide-to-plotting-data-in-python/plot-spatial-data/customize-vector-plots/python-customize-map-legends-geopandas/

ax = df.plot(figsize=(12,12), color='whitesmoke', linestyle=':', edgecolor='black')

df_list_geocoded.plot(markersize=1, ax=ax,

                column='room_type',

                categorical=True,

                legend=True)
long_median = df_list['longitude'].median()

lat_median = df_list['latitude'].median()

(lat_median,long_median)
import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
m_1 = folium.Map(location=[lat_median,long_median], tiles='openstreetmap', zoom_start=10)

m_1
folium.Map(

    location=[lat_median,long_median],

    tiles='Stamen Toner',

    zoom_start=13

)




m = folium.Map(

    location=[lat_median,long_median],

    tiles='Stamen Terrain',

    zoom_start=13

)



m.add_child(folium.LatLngPopup())



m



df_list['room_type'].value_counts()
df_list['amenities'].str.lower().str.contains('cooking').sum()
df_list_cooking = df_list[df_list['amenities'].str.lower().str.contains('cooking')]
df_list_cooking_wifi = df_list_cooking[df_list_cooking['amenities'].str.lower().str.contains('wifi')]
df_hotel_cooking_wifi = df_list_cooking_wifi[df_list_cooking_wifi['room_type']=='Hotel room']
m_2 = folium.Map(location=[lat_median,long_median], tiles='cartodbpositron', zoom_start=13)





for idx, row in df_hotel_cooking_wifi.iterrows():

    Marker([row['latitude'], row['longitude']], popup=row['name']).add_to(m_2)



m_2
df_calendar = pd.read_csv('/kaggle/input/munich-airbnb-listings-data-may-24-2020/calendar.csv')
df_calendar.info()
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
df_calendar['price'] = df_calendar['price'].str.replace('$', '')
df_calendar['price'] = df_calendar['price'].str.replace(',', '')
df_calendar['price'] = pd.to_numeric(df_calendar['price'])
df_calendar['price'].hist()
df_calendar.groupby('date')['price'].median()
df_calendar.groupby('date')['price'].median().plot()
# https://github.com/stefan-jansen/machine-learning-for-trading/blob/master/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/03_normalize_tick_data.ipynb

# https://www.packtpub.com/in/data/machine-learning-for-algorithmic-trading-second-edition



import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(15, 8))

df_calendar.groupby('date')['price'].median().plot(ax=axes[0])

df_calendar.groupby('date')['minimum_nights'].median().plot(ax=axes[1])

df_calendar.groupby('date')['maximum_nights'].median().plot(ax=axes[2])



axes[0].set_title('price', fontsize=14)

axes[1].set_title('minimum_nights', fontsize=14)

axes[2].set_title('maximum_nights', fontsize=14)

fig.autofmt_xdate()

fig.suptitle('Airbnb data')

fig.tight_layout()

plt.subplots_adjust(top=0.9)



df_covid = pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')

df_covid.info()
df_covid['county'].unique()
df_covid['county'].value_counts()
df_covid['date'] = pd.to_datetime(df_covid['date'])
df_covid
# https://github.com/stefan-jansen/machine-learning-for-trading/blob/master/02_market_and_fundamental_data/01_NASDAQ_TotalView-ITCH_Order_Book/03_normalize_tick_data.ipynb

# https://www.packtpub.com/in/data/machine-learning-for-algorithmic-trading-second-edition



import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(15, 8))

df_covid.groupby('date')['cases'].sum().plot(ax=axes[0])

df_covid.groupby('date')['deaths'].median().plot(ax=axes[1])

df_covid.groupby('date')['recovered'].median().plot(ax=axes[2])



axes[0].set_title('cases', fontsize=14)

axes[1].set_title('deaths', fontsize=14)

axes[2].set_title('recovered', fontsize=14)

fig.autofmt_xdate()

fig.suptitle('Covid data')

fig.tight_layout()

plt.subplots_adjust(top=0.9)


