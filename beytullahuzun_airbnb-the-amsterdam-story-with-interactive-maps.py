import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import folium

from folium.plugins import FastMarkerCluster

import geopandas as gpd

from branca.colormap import LinearColormap

import os
print(os.listdir("../input"))
listings = pd.read_csv("../input/listings.csv")

listings_details = pd.read_csv("../input/listings_details.csv", low_memory=False)

print(listings.shape)
listings.columns
target_columns = ["id", "property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin", "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time", "host_response_rate", "street", "weekly_price", "monthly_price", "market"]

listings = pd.merge(listings, listings_details[target_columns], on='id', how='left')

listings.info()
listings = listings.drop(columns=['neighbourhood_group'])

listings['host_response_rate'] = pd.to_numeric(listings['host_response_rate'].str.strip('%'))



listings.head()
feq=listings['neighbourhood'].value_counts().sort_values(ascending=True)

feq.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Number of listings by neighbourhood", fontsize=18)

plt.xlabel('Number of listings', fontsize=10)

plt.show()
#An alternative to the bar chart above with a few changes

feq=listings['neighbourhood'].value_counts().sort_values(ascending=True)

feq.plot.barh(figsize=(10, 8), width=2)

plt.title("Number of listings by neighbourhood (2)", fontsize=20)

plt.xlabel('Number of listings', fontsize=12)



plt.show()
lats2018 = listings['latitude'].tolist()

lons2018 = listings['longitude'].tolist()

locations = list(zip(lats2018, lons2018))



map1 = folium.Map(location=[52.3680, 4.9036], zoom_start=11.5)

FastMarkerCluster(data=locations).add_to(map1)

map1
freq = listings['room_type']. value_counts().sort_values(ascending=True)

freq.plot.barh(figsize=(15, 3), width=1, color = ["g","b","r"])

plt.show()
listings.property_type.unique()
prop = listings.groupby(['property_type','room_type']).room_type.count()

prop = prop.unstack()

prop['total'] = prop.iloc[:,0:3].sum(axis = 1)

prop = prop.sort_values(by=['total'])

prop = prop[prop['total']>=100]

prop = prop.drop(columns=['total'])



prop.plot(kind='barh',stacked=True, color = ["r","b","g"],

              linewidth = 1, grid=True, figsize=(15,8), width=1)

plt.title('Property types in Amsterdam', fontsize=18)

plt.xlabel('Number of listings', fontsize=14)

plt.ylabel("")

plt.legend(loc = 4,prop = {"size" : 13})

plt.rc('ytick', labelsize=13)

plt.show()
feq=listings['accommodates'].value_counts().sort_index()

feq.plot.bar(figsize=(10, 8), color='b', width=1, rot=0)

plt.title("Accommodates (number of people)", fontsize=20)

plt.ylabel('Number of listings', fontsize=12)

plt.xlabel('Accommodates', fontsize=12)

plt.show()
private = listings[listings['room_type'] == "Private room"]

host_private = private.groupby(['host_id', 'host_name', 'street']).size().reset_index(name='private_rooms').sort_values(by=['private_rooms'], ascending=False)

host_private.head()
feliciano = private[private['host_id']== 67005410]

feliciano = feliciano[['id', 'name','host_name', 'latitude', 'longitude']]

feliciano
freq = listings.groupby(['host_id']).size().reset_index(name='num_host_listings')

host_prop = freq.groupby(['num_host_listings']).size().reset_index(name='count').transpose()

host_prop.columns = host_prop.iloc[0]

host_prop = host_prop.drop(host_prop.index[0])

host_prop



freq = listings.groupby(['host_id', 'host_name', 'host_about']).size().reset_index(name='num_host_listings')

freq = freq.sort_values(by=['num_host_listings'], ascending=False)

freq = freq[freq['num_host_listings'] >= 20]

freq
feq = listings[listings['accommodates']==2]

feq = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)

feq.plot.barh(figsize=(10, 8), color='b', width=1)

plt.title("Average daily price for a 2-persons accommodation", fontsize=20)

plt.xlabel('Average daily price (Euro)', fontsize=12)

plt.ylabel("")

plt.show()
adam = gpd.read_file("../input/neighbourhoods.geojson")

feq = pd.DataFrame([feq])

feq = feq.transpose()

adam = pd.merge(adam, feq, on='neighbourhood', how='left')

adam.rename(columns={'price': 'average_price'}, inplace=True)

adam.average_price = adam.average_price.round(decimals=0)



map_dict = adam.set_index('neighbourhood')['average_price'].to_dict()

color_scale = LinearColormap(['yellow','red'], vmin = min(map_dict.values()), vmax = max(map_dict.values()))



def get_color(feature):

    value = map_dict.get(feature['properties']['neighbourhood'])

    return color_scale(value)



map3 = folium.Map(location=[52.3680, 4.9036], zoom_start=11)

folium.GeoJson(data=adam,

               name='Amsterdam',

               tooltip=folium.features.GeoJsonTooltip(fields=['neighbourhood', 'average_price'],

                                                      labels=True,

                                                      sticky=False),

               style_function= lambda feature: {

                   'fillColor': get_color(feature),

                   'color': 'black',

                   'weight': 1,

                   'dashArray': '5, 5',

                   'fillOpacity':0.5

                   },

               highlight_function=lambda feature: {'weight':3, 'fillColor': get_color(feature), 'fillOpacity': 0.8}).add_to(map3)

map3
fig = plt.figure(figsize=(20,10))

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=20)



ax1 = fig.add_subplot(121)

feq = listings[listings['number_of_reviews']>=10]

feq1 = feq.groupby('neighbourhood')['review_scores_location'].mean().sort_values(ascending=True)

ax1=feq1.plot.barh(color='b', width=1)

plt.title("Average review score location (at least 10 reviews)", fontsize=20)

plt.xlabel('Score (scale 1-10)', fontsize=20)

plt.ylabel("")



ax2 = fig.add_subplot(122)

feq = listings[listings['accommodates']==2]

feq2 = feq.groupby('neighbourhood')['price'].mean().sort_values(ascending=True)

ax2=feq2.plot.barh(color='b', width=1)

plt.title("Average daily price for a 2-persons accommodation", fontsize=20)

plt.xlabel('Average daily price (Euro)', fontsize=20)

plt.ylabel("")



plt.tight_layout()

plt.show()
listings10 = listings[listings['number_of_reviews']>=10]



fig = plt.figure(figsize=(20,15))

plt.rc('xtick', labelsize=16) 

plt.rc('ytick', labelsize=16)



ax1 = fig.add_subplot(321)

feq=listings10['review_scores_location'].value_counts().sort_index()

ax1=feq.plot.bar(color='b', width=1, rot=0)

#ax1.tick_params(axis = 'both', labelsize = 16)

plt.title("Location", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



ax2 = fig.add_subplot(322)

feq=listings10['review_scores_cleanliness'].value_counts().sort_index()

ax2=feq.plot.bar(color='b', width=1, rot=0)

plt.title("Cleanliness", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



ax3 = fig.add_subplot(323)

feq=listings10['review_scores_value'].value_counts().sort_index()

ax3=feq.plot.bar(color='b', width=1, rot=0)

plt.title("Value", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



ax4 = fig.add_subplot(324)

feq=listings10['review_scores_communication'].value_counts().sort_index()

ax4=feq.plot.bar(color='b', width=1, rot=0)

plt.title("Communication", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



ax5 = fig.add_subplot(325)

feq=listings10['review_scores_checkin'].value_counts().sort_index()

ax5=feq.plot.bar(color='b', width=1, rot=0)

plt.title("Arrival", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



ax6 = fig.add_subplot(326)

feq=listings10['review_scores_accuracy'].value_counts().sort_index()

ax6=feq.plot.bar(color='b', width=1, rot=0)

plt.title("Accuracy", fontsize=24)

plt.ylabel('Number of listings', fontsize=14)

plt.xlabel('Average review score', fontsize=14)



plt.tight_layout()

plt.show()
feq=listings['host_is_superhost'].value_counts()

feq.plot.bar(figsize=(10, 8), width=1, rot=0)

plt.title("Number of listings with Superhost", fontsize=20)

plt.ylabel('Number of listings', fontsize=12)

plt.show()
fig = plt.figure(figsize=(20,10))

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=20)



ax1 = fig.add_subplot(121)

feq1 = listings10['host_response_rate'].dropna()

ax1= plt.hist(feq1)

plt.title("Response rate (at least 10 reviews)", fontsize=20)

plt.ylabel("number of listings")

plt.xlabel("percent", fontsize=20)



ax2 = fig.add_subplot(122)

feq2 = listings10['host_response_time'].value_counts()

ax2=feq2.plot.bar(color='b', width=1, rot=45)

plt.title("Response time (at least 10 reviews)", fontsize=20)

plt.ylabel("number of listings")



plt.tight_layout()

plt.show()
listings.host_response_rate.unique()
listings['host_response_rate'].max()

feq1
listings['review_scores_cleanliness'].value_counts()
#feq = listings[listings['number_of_reviews']>=10]

feq = listings

feq1 = feq.groupby('neighbourhood')['review_scores_location'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

feq1