#dealing with data

import numpy as np

import pandas as pd



#plot

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns



#map

import geopandas as gpd

from shapely import wkt



#wordcloud

from wordcloud import WordCloud
airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')



airbnb.head(5)
airbnb.shape
airbnb.isnull().sum()
airbnb['price'].quantile([0,.25, .5,0.75,1])
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=airbnb,y='price')

plt.ylim(0,1000)
#Number of 0$

len(airbnb[airbnb.price==0.0])
#Number of 10,000$

len(airbnb[airbnb.price==10000.0])
len(airbnb[airbnb.price>2000.0])
#Remove 0$ and take only price under 2000$ by night

airbnb = airbnb[airbnb.price != 0.0]

airbnb = airbnb[airbnb.price <= 2000.0]
sns.countplot(airbnb['room_type'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Number of listings by type of room ')

# Data for maps

nyc = gpd.read_file(gpd.datasets.get_path('nybb'))

nyc.head(5)



# Count the number of listings by borough

borough_count = airbnb.groupby('neighbourhood_group').agg('count').reset_index()



#Rename the column to join the data 

nyc.rename(columns={'BoroName':'neighbourhood_group'}, inplace=True)

bc_geo = nyc.merge(borough_count, on='neighbourhood_group')



#Plot the count by borough into a map

fig,ax = plt.subplots(1,1, figsize=(10,10))

bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax, legend=True)

bc_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)

plt.title("Number of Airbnb Listings by NYC Borough")

plt.axis('off')
sns.countplot(airbnb['neighbourhood_group'], palette="plasma")

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Number of listings by borough ')

# Data group by neighbourhood_group (borough)

airbnb_neighbourhood_group = airbnb.groupby(['neighbourhood_group']) 

# Quartile by borough

airbnb_neighbourhood_group['price'].quantile([0,.25, .5,0.75,1]).to_frame()
viz_price_neighbourhood_group=sns.violinplot(data=airbnb[airbnb.price < 500], x='neighbourhood_group', y='price')

viz_price_neighbourhood_group.set_title('Density and distribution of prices for each borough')
#Compute average price by borough and join

borough_price = airbnb.groupby('neighbourhood_group').agg('median').reset_index()[['neighbourhood_group','price']]

bp_geo = nyc.merge(borough_price, on='neighbourhood_group')

bp_geo
fig,ax = plt.subplots(1,1, figsize=(10,10))

bp_geo.plot(column='price', cmap='plasma_r', alpha=.5, ax=ax, legend=True) #change cmap for colors

bp_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)

plt.title("Median price of Airbnb by NYC Borough")

plt.axis('off')
plt.figure(figsize=(10,10))

ax = sns.boxplot(data=airbnb[airbnb.neighbourhood_group == "Manhattan"],y='price')

plt.ylim(0,1000)
airbnb_manhattan = airbnb[airbnb.neighbourhood_group == "Manhattan"].groupby(['neighbourhood']) 

airbnb_med_manhattan = airbnb_manhattan['price'].median().to_frame()



airbnb_med_manhattan_sort = airbnb_med_manhattan.sort_values(by='price')

plt.figure(figsize=(40,30))

ax = sns.barplot(x=airbnb_med_manhattan_sort.index, y="price", data=airbnb_med_manhattan_sort)

plt.figure(figsize=(10,10))

ax = sns.boxplot(data=airbnb[airbnb.neighbourhood_group == "Bronx"],y='price')

plt.ylim(0,1000)
airbnb_bronx = airbnb[airbnb.neighbourhood_group == "Bronx"].groupby(['neighbourhood']) 

airbnb_bronx_med = airbnb_bronx['price'].median().to_frame()





airbnb_bronx_med_sort = airbnb_bronx_med.sort_values(by='price')

plt.figure(figsize=(40,30))

ax = sns.barplot(x=airbnb_bronx_med_sort.index, y="price", data=airbnb_bronx_med_sort)
airbnb.groupby(['neighbourhood'])['price'].median().to_frame().sort_values(by='price')
# Import data from the website (find it)

nbhoods = pd.read_csv('../input/nyntacsv/nynta.csv')

nbhoods.head(5)
#Rename the column

nbhoods.rename(columns={'NTAName':'neighbourhood'}, inplace=True)



#Convert the geometry column text into well known text (librairy shapely)

nbhoods['geom'] = nbhoods['the_geom'].apply(wkt.loads)



#Now convert the pandas dataframe into a Geopandas GeoDataFrame

nbhoods = gpd.GeoDataFrame(nbhoods, geometry='geom')
airbnb = gpd.GeoDataFrame(airbnb, geometry=gpd.points_from_xy(airbnb.longitude, airbnb.latitude))



# Spatial join (this code runs an intersect analysis to find which neighborhood the Airbnb location is in)

joined = gpd.sjoin(nbhoods, airbnb, how='inner', op='intersects')

joined.drop(columns='geom', inplace=True)

joined.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)

nb_join_price = joined.groupby('neighbourhood').agg('median').reset_index()[['neighbourhood','price']]

true_count = nbhoods.merge(nb_join_price, on='neighbourhood')

fig,ax = plt.subplots(1,1, figsize=(10,10))



base = nbhoods.plot(color='white', edgecolor='black', ax=ax)



true_count.plot(column='price',cmap='plasma_r', ax=base, legend=True)

plt.title('Median Price of listings by Neighborhood in NYC')
# Global wordcloud

name = " ".join(str(w) for w in airbnb.name)

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080,max_words=60

                         ).generate(name)

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('name.png')

plt.show()
name_manhattan = " ".join(str(w) for w in airbnb.name[airbnb.neighbourhood_group == "Manhattan"])

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080,max_words=30

                         ).generate(name_manhattan)

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('name.png')

plt.show()
name_bronx = " ".join(str(w) for w in airbnb.name[airbnb.neighbourhood_group == "Bronx"])

plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=1920,

                          height=1080,max_words=30

                         ).generate(name_bronx)

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('name.png')

plt.show()