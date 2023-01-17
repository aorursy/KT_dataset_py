import pandas as pd



import os

print(os.listdir("../input"))
import seaborn as sns

import matplotlib.pyplot as plt
customer = pd.read_csv("../input/olist_customers_dataset.csv")

customer.head(0)
print ("Number of unique cities is ",customer['customer_city'].nunique())

print ("Number of unique zip codes is ",customer['customer_zip_code_prefix'].nunique())

print ("Number of unique states is ",customer['customer_state'].nunique())
customer['customer_state'].unique()


plt.figure(figsize=(20,6))

sns.set(style='darkgrid')



ax = sns.countplot(x="customer_state", data=customer, order=customer['customer_state'].value_counts().index)

ax.set(xlabel='States', ylabel='Customer count')



plt.title("States and their customer base")

plt.show()



plt.figure(figsize=(20,6))

sns.set(style='darkgrid')



ax = sns.countplot(x="customer_city", data=customer, order=customer['customer_city'].value_counts().index[0:10])

ax.set(xlabel='City', ylabel='Customer count')



plt.title("Top 10 cities with highest customer base")

plt.show()

geo_data = pd.read_csv("../input/olist_geolocation_dataset.csv")



print (list(geo_data.columns.values))

geo_data.rename(columns = {"geolocation_zip_code_prefix":"zip_code",

                         "geolocation_lat": "latitude", 

                         "geolocation_lng":"longitude",

                         "geolocation_city":"city",

                         "geolocation_state":"state"               

                        },inplace = True)

print (list(geo_data.columns.values))
#Removing some outliers



#Brazils most Northern spot is at 5 deg 16′ 27.8″ N latitude.;

geo_data = geo_data[geo_data.latitude <= 5.27438888]

#it’s most Western spot is at 73 deg, 58′ 58.19″W Long.

geo_data = geo_data[geo_data.longitude >= -73.98283055]

#It’s most southern spot is at 33 deg, 45′ 04.21″ S Latitude.

geo_data = geo_data[geo_data.latitude >= -33.75116944]

#It’s most Eastern spot is 34 deg, 47′ 35.33″ W Long.

geo_data = geo_data[geo_data.longitude <=  -34.79314722]
import geopandas as gpd

import descartes

from shapely.geometry import Point, Polygon
geometry = [Point(xy) for xy in zip(geo_data['latitude'],geo_data['longitude'])]
crs = {'init':'espg:4326'}

geo_df = gpd.GeoDataFrame(geo_data, geometry = geometry, crs=crs)

geo_df.head(2)
plt.figure(figsize=(10,10))



cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))



ax = cities.plot(color='white', edgecolor='black')



geo_df.plot(ax=ax, color='red')



plt.show()



#Reference: http://geopandas.org/index.html

#Updates coming soon

#Requires updates:

#Making the graph more readable and interpretable

#By making it bigger, one can find the area which is more developed [ Those places will have more customers ]
order_item = pd.read_csv("../input/olist_order_items_dataset.csv")

#print (list(order_item.columns.values))

order_item.rename(columns = {"shipping_limit_date":"shipping_date"},inplace = True)

print (list(order_item.columns.values))



#print (order_item.dtypes)



order_item.sample(10)

order_item['order_item_id'].value_counts()
import scipy.stats as st
price = order_item.price
plt.figure(figsize=(20,6))

fig, axs = plt.subplots(nrows=2)



sns.boxplot(price, ax=axs[0])

sns.distplot(price, fit=st.norm, ax=axs[1])



plt.title("Distribution of price")

plt.show()

import numpy as np
print ("Mean price of item is ",np.mean(order_item.price))

print ("Standard deviation of price is ",np.std(order_item.price))
print (price.describe())


price = price[price <= price.quantile([.75])[0.75]]
fig, axs = plt.subplots(ncols=3)

fig.set_size_inches(20,4)



sns.boxplot(price, ax=axs[0])

axs[0].set_title("Box plot of price")



sns.distplot(price, fit=st.norm, ax=axs[1])

axs[1].set_title("Fitting normal distribution to price")



sns.distplot(price, fit=st.gamma, ax=axs[2])

axs[2].set_title("Fitting Gamma distribution to price")



plt.show()

date_time = order_item.shipping_date.str.split(' ')



#To be continued