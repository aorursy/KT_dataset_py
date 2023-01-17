# import libraries

# ================



# data processing

import numpy as np

import pandas as pd

import geopandas as gpd



# visualizations

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import folium
# palette

pal = ['#EF585D', # red

       '#016E6F', # green

       '#494949', # dark-grey

       '#8C1845', # purple

       '#FFB401', # yellow

       '#D93801', # saffron

       '#EBEBEB'] # light-grey

sns.set_style("whitegrid")
# plotly offline

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
# import dataset

df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
# columns

df.columns
# info

# df.info()
# datatype

# df.dtype
# describe

# df.describe(include='all')
# shape

# df.shape
# no. of missing values

df.isna().sum()
# no. of unique values

df.nunique()
# value counts 

print(df['neighbourhood_group'].value_counts())

print('')

print(df['room_type'].value_counts())
fig, axes = plt.subplots(1, 2, figsize=(20, 6))



sns.scatterplot(data=df, x='longitude', y='latitude', hue='neighbourhood_group', palette='Dark2', ax=axes[0])

sns.scatterplot(data=df, x='longitude', y='latitude', hue='neighbourhood', legend=False, palette='Dark2', ax=axes[1])



axes[0].set_title('Neighbourhood Group')

axes[1].set_title('Neighbourhood')

axes[0].set_ylabel('')

axes[0].set_xlabel('')

axes[1].set_ylabel('')

axes[1].set_xlabel('')



plt.show()
fig, axes = plt.subplots(2, 3, figsize=(20, 9))



axes[0,0].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['price']), cmap='RdPu', alpha=0.8)

axes[0,0].set_title('Neighbourhood Group')



axes[0,1].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['reviews_per_month']), cmap='GnBu', alpha=0.8)

axes[0,1].set_title('Reviews per month')



axes[0,2].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['availability_365']), cmap='YlGn', alpha=0.6)

axes[0,2].set_title('Availability 365')



axes[1,0].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['number_of_reviews']), cmap='RdPu', alpha=0.6)

axes[1,0].set_title('Number of reviews')



axes[1,1].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['minimum_nights']), cmap='GnBu', alpha=0.6)

axes[1,1].set_title('Minimum Nights')



axes[1,2].scatter(x=df['longitude'], y=df['latitude'], c=np.log(df['calculated_host_listings_count']), cmap='YlGn', alpha=0.6)

axes[1,2].set_title('Host listing count Group')



plt.show()
# costliest neighbourhood_group

temp = df.groupby(['neighbourhood_group'])['price'].mean().reset_index()

temp = temp.sort_values('price', ascending=False).reset_index(drop=True)



ax = sns.barplot(data=temp, x='price', y='neighbourhood_group', palette='Dark2')

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Mean price from each neighbourhood_group')

plt.show()
# costliest neighbourhood

temp = df.groupby(['neighbourhood_group', 'neighbourhood'])['price'].agg(['mean', 'count']).reset_index()

temp = temp.sort_values('mean', ascending=False).reset_index(drop=True)

# temp.head()



#plot

plt.figure(figsize=(6, 6))

ax = sns.barplot(data=temp.head(20), x='mean', y='neighbourhood', 

            hue='neighbourhood_group', palette='Dark2', dodge=False)

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Mean price from each neighbourhood')

plt.show()
# temp = df[['neighbourhood', 'price', 'neighbourhood_group']]

# temp = temp.groupby('neighbourhood_group', 'neighbourhood').reset_index()



plt.figure(figsize=(6, 50))

ax = sns.stripplot(data=df, y='neighbourhood', x='price', hue='neighbourhood_group', palette='Dark2')

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Price Distribution based on neighbourhood group')

plt.show()
ax = sns.stripplot(data=df, x='neighbourhood_group', y='price', palette='Dark2')

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Price Distribution based on neighbourhood group')

plt.show()
ax = sns.stripplot(data=df, x='room_type', y='price', palette='Dark2')

ax.set_ylabel('')

ax.set_xlabel('')

ax.set_title('Price Distribution based on room type')

plt.show()
plt.figure(figsize=(8, 4))

df_pivot = df.pivot_table(values='price', index='room_type', columns='neighbourhood_group', aggfunc='mean')

sns.heatmap(df_pivot, annot=True, fmt='.1f', cmap='Purples')

plt.suptitle('Mean Price')

plt.xlabel('')

plt.ylabel('')

plt.show()
plt.figure(figsize=(8, 4))

df_pivot = df.pivot_table(values='reviews_per_month', index='room_type', columns='neighbourhood_group', aggfunc='mean')

sns.heatmap(df_pivot, annot=True, fmt='.2f', cmap='Blues')

plt.suptitle('Mean no. of reviews in a month')

plt.xlabel('')

plt.ylabel('')

plt.show()
plt.figure(figsize=(8, 4))

df_pivot = df.pivot_table(values='host_id', index='room_type', columns='neighbourhood_group', aggfunc='count')

sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='Greens')

plt.suptitle('No. of hosts')

plt.xlabel('')

plt.ylabel('')

plt.show()
# df.columns
# cols = ['neighbourhood_group', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count']

# sns.pairplot(df[cols], hue="neighbourhood_group")
# cols = ['room_type', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 'reviews_per_month', 'calculated_host_listings_count']

# sns.pairplot(df[cols], hue="room_type")
# cols = ['room_type', 'price', 'minimum_nights', 'number_of_reviews', 'last_review', 

#         'reviews_per_month', 'calculated_host_listings_count', 'availability_365']



# plt.figure(figsize=(8, 6))

# df_corr = df[cols].corr()

# sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdBu', vmax=0.8, vmin=-0.8)

# plt.show()
# plt.figure(figsize=(15, 6))



# for val in df['neighbourhood_group'].unique():

#     ax = sns.kdeplot(df[df['neighbourhood_group']==val]['price'], label=val, shade=True)



# ax.set_xlabel('Price')

# ax.set_title('Distribution of Price')

# plt.figure
# plt.figure(figsize=(15, 6))



# for val in df['room_type'].unique():

#     ax = sns.kdeplot(df[df['room_type']==val]['price'], label=val, shade=True)



# ax.set_xlabel('Price')

# ax.set_title('Distribution of Price')

# plt.figure
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='minimum_nights')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='price')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='number_of_reviews')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='reviews_per_month')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='calculated_host_listings_count')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='availability_365')
# sns.scatterplot(data=df, x='latitude', y='longitude', hue='room_type')
# for col in ['neighbourhood_group', 'room_type', 'price', 'minimum_nights', 'number_of_reviews', 

#             'reviews_per_month', 'calculated_host_listings_count', 'availability_365']:

#     plt.figure(figsize=(10,6))

#     sns.scatterplot(data=df, x='latitude', y='longitude', hue=col)

#     plt.plot()
nyc_map = gpd.read_file('../input/new-york-shapefile-16/cb_2016_36_tract_500k.shp')

nyc_map.head()
# nyc_map['AWATER'].value_counts()
# nyc_map = gpd.read_file('../input/new-york-shapefile-16/cb_2016_36_tract_500k.shp')

# bnb_loc = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))



# nyc_map.crs = {'init' :'epsg:4326'}

# bnb_loc.crs = {'init' :'epsg:4326'}



# nyc_bnb = gpd.sjoin(nyc_map, bnb_loc)
# m = folium.Map(location=[40.7, -73.95], tiles='cartodbpositron',

#                min_zoom=8, max_zoom=8, zoom_start=8)



# for i in range(0, len(df)):

#     folium.Circle(

#         location=[df.iloc[i]['latitude'], df.iloc[i]['longitude']],

#         color='crimson',

#         radius=1).add_to(m)

    

# m
# fig, ax = plt.subplots(figsize=(12, 6))

# map_joined.plot(ax=ax, color='lightgrey')

# gdf.plot(ax=ax, column='price', markersize=1, cmap='plasma')

# ax.set_axis_off()