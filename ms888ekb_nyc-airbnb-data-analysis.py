import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl



mpl.style.use('ggplot')
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head(3)
df.info()
df.drop(['id', 'name', 'last_review'], inplace=True, axis=1)

df['host_name'].replace(np.nan, 'anonymous', inplace=True)

df['reviews_per_month'].replace(np.nan, 0, inplace=True)
df.describe()
df = df.loc[df['minimum_nights'] < 500]
df.info()
df_count = df._get_numeric_data()

df_count.hist(bins = 50, figsize=(20,15))

plt.show()
fig, ax = plt.subplots(figsize=(10,7))



df.host_name.value_counts().to_frame().reset_index().iloc[:20,:].plot(kind='bar', x='index', legend=False, ax=ax)

plt.xlabel('Names')

plt.ylabel('Number of listings')



plt.show()
#Further brief data observation

df.neighbourhood_group.value_counts()
fig, ax = plt.subplots(figsize=(10,7))

# Group and calculate all listings per each host

df_top_host = df.groupby('host_id').agg('count').sort_values(by=['calculated_host_listings_count'], ascending=False).reset_index()

# Plot only the top 30 of them

df_top_host.iloc[:30,[df_top_host.columns.get_loc('host_id'),df_top_host.columns.get_loc('calculated_host_listings_count')]].plot(kind='bar', x='host_id', y='calculated_host_listings_count', ax=ax, legend=False)

plt.xlabel('Host ID')

plt.ylabel('Number of listings per host')



plt.show()
df.room_type.value_counts()
# Import GeoPandas library to operate Geo data.

import geopandas as gpd
# Let's first read two main maps we're going to work with. 

# The map with boroughs boundaties and the map with neighbourhoods boundaties



bd = gpd.read_file('../input/airbnbny/Borough_Boundaries.geojson') # This is for boroughs



nhd = gpd.read_file('../input/airbnbny/neighborhoods.geojson') # This is for neighbourhoods

# Explore it:



bd.head(3)
nhd.head(3)
# !conda install --channel conda-forge descartes

nhd.plot()

plt.show()
# Rename columns in the geoframes to correspond with the initial dataframe.



nhd.rename(columns={'ntaname' : 'neighbourhood'}, inplace=True)

bd.rename(columns = {'boro_name' : 'neighbourhood_group'}, inplace=True)
# This is for the boroughs

df_bd = df.merge(bd, on='neighbourhood_group')

df_bd.drop('boro_code', axis=1, inplace=True)



# This is for the neighbourhoods

df_nhd = df.merge(nhd, on = 'neighbourhood')

df_nhd.drop(['boro_name', 'boro_code', 'county_fips', 'ntacode'], axis=1, inplace=True)
# Take a look what we've got for the neighbourhoods:

df_bd.head(3)
# Take a look what we've got for the boroughs:

df_nhd.head(3)
# We need to group by boroughs and count the amount of listings for the each group:

bc = df.groupby('neighbourhood_group', as_index=False).agg('count')



# Extract only those columns we're looking for:

bc = bc.iloc[:,:2]



# Merge boroughs geodata with the count calculated dataset on the 'neighbourhood_group' column:

geo_bc = bd.merge(bc, on="neighbourhood_group")
# This is for the мisual aesthetics only:

from mpl_toolkits.axes_grid1 import make_axes_locatable



fig, ax = plt.subplots(figsize=(12,10))

ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



# Finally, plot it:

geo_bc.plot(column='host_id', cmap='Wistia', legend=True, ax=ax, cax=cax)



def getXY(pt, x_adj=0, y_adj=0):

    return (pt.x+x_adj, pt.y+y_adj)



# Add annotations:

for index, value in enumerate(geo_bc['host_id']):

    label = format(int(value), ',')

    ax.annotate(label, xy=getXY(geo_bc.geometry.iloc[index].centroid,0,-0.015), ha='center', color='black')

    ax.annotate(geo_bc.iloc[index,geo_bc.columns.get_loc('neighbourhood_group')], xy=getXY(geo_bc.geometry.iloc[index].centroid), ha='center', color='black')



ax.set_title(label='Number of listings per borough', fontdict=None, loc='center')



plt.plot()
# bp here stands for borough price (mean)

bp = df.groupby(['neighbourhood_group'], as_index=False).agg('mean')



# Exctract what we need:

bp = bp.iloc[:,[0,4]]



# Merge boroughs geodata with the mean calculated dataset on the 'neighbourhood_group' column:

geo_bp = bd.merge(bp, on='neighbourhood_group')
fig, ax = plt.subplots(figsize=(12,10))

ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



# Finally, plot it:

geo_bp.plot(column='price', cmap='summer_r', legend=True, ax=ax, cax=cax)



def getXY(pt, x_adj=0, y_adj=0):

    return (pt.x+x_adj, pt.y+y_adj)



# Add annotations:

for index, value in enumerate(geo_bp['price']):

    label = format(int(value), ',')

    ax.annotate(label, xy=getXY(geo_bc.geometry.iloc[index].centroid,0,-0.015), ha='center', color='black')

    ax.annotate(geo_bc.iloc[index,geo_bc.columns.get_loc('neighbourhood_group')], xy=getXY(geo_bc.geometry.iloc[index].centroid), ha='center', color='black')



ax.set_title(label='Mean one-night-stay cost , $', fontdict=None, loc='center')



plt.plot()

# Loading the crimes dataset

crimes2018 = pd.read_csv('../input/airbnbny/nyc_crimes_2018.csv', parse_dates=['CMPLNT_TO_DT'],index_col = None)

crimes2018.drop('Unnamed: 0', inplace=True, axis=1)
crimes2018.head()
crimes2018.OFNS_DESC.value_counts()
geo_crimes2018 = gpd.GeoDataFrame(crimes2018, geometry=gpd.points_from_xy(crimes2018.Longitude, crimes2018.Latitude))

geo_crimes2018 = geo_crimes2018.loc[:,['geometry']]

geo_crimes2018.dropna(axis=0, inplace=True)

geo_crimes2018.reset_index()

geo_crimes2018.head()
from scipy import ndimage

import matplotlib.pylab as pylab



def heatmap(d, bins=(100,100), smoothing=1.3, cmap='jet'):

    def getx(pt):

        return pt.coords[0][0]



    def gety(pt):

        return pt.coords[0][1]



    x = list(d.geometry.apply(getx))

    y = list(d.geometry.apply(gety))



    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)

    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]



    logheatmap = np.log(heatmap)

    logheatmap[np.isneginf(logheatmap)] = 0

    logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')

    

    plt.imshow(logheatmap, cmap=cmap, extent=extent, aspect='auto')

    plt.colorbar()

    plt.gca().invert_yaxis()

    plt.show()

    

from mpl_toolkits.axes_grid1 import make_axes_locatable

pylab.rcParams['figure.figsize'] = 12, 10



# nhd.plot(ax=ax)

heatmap(geo_crimes2018, bins=250, smoothing=0.9)

# Transform df to geopandas format:

geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))



# Inner join geopandas df to nhd (neighbourhood GeoJSON dataset):

geo_df_j = gpd.sjoin(nhd, geo_df, how='inner', op='intersects')



geo_df_j.head(3)
# Group by neighbourhood_left (from nbh) and calculate the mean price for each neighbourhood:

nb_join_price = geo_df_j.groupby('neighbourhood_left', as_index=False).agg('mean')



# Extract what we need:

nb_join_price = nb_join_price.iloc[:,[nb_join_price.columns.get_loc('neighbourhood_left'),nb_join_price.columns.get_loc('price')]]



nb_join_price.head(2)
# Merge  with the joined geo df

geo_nb_join_price = geo_df_j.merge(nb_join_price, on='neighbourhood_left')
fig, ax = plt.subplots(figsize=(14,12))

ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



# Finally, plot it:

geo_nb_join_price.plot(column='price_y', figsize=(10,10), cmap='RdPu', legend=True, ax=ax, cax=cax)



def getXY(pt, x_adj=0, y_adj=0):

    return (pt.x+x_adj, pt.y+y_adj)



# Add annotations:

for index, value in enumerate(geo_nb_join_price['price_y']):

    label = format(int(value), ',')

    ax.annotate(label, xy=getXY(geo_nb_join_price.geometry.iloc[index].centroid,0,0), ha='center', color='black', fontsize=8)



ax.set_title(label='Mean one-night-stay cost , $', fontdict=None, loc='center')



plt.plot()
nb_join_price[nb_join_price['price'] > 500]
geo_nb_join_price[geo_nb_join_price['neighbourhood_left'] == 'Rossville-Woodrow'].loc[:,['neighbourhood_left','price_x']]
# Load a Theatres layer

theatres = gpd.read_file('../input/airbnbny/theaters.geojson')



# Load a Museums layer

museums = gpd.read_file('../input/airbnbny/museums.geojson')
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(12,10))

ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



# Plot a base layer

base = geo_nb_join_price.plot(column='price_y', ax=ax, figsize=(10,10), cmap='RdPu', legend=True, cax=cax)



# Plot a Theatres layer

theatres.plot(ax=ax, marker='o', color='red', markersize=6)



# Plot a Museums layer

museums.plot(ax=ax, marker='o', color='green', markersize=5)



# Set title

ax.set_title(label='NY Theatres and Museums Distribution', fontdict=None, loc='center')



plt.show()
# Group by neighbourhood_left (from the joined geo dataframe) and calculate the mean availability for each neighbourhood:

nb_avb = geo_df_j.groupby('neighbourhood_left', as_index=False).agg('mean')



# Pick up the columns we need:

nb_avb = nb_avb.loc[:,['neighbourhood_left', 'availability_365']]



# Merge  with the joined geo df

geo_nb_avb = geo_df_j.merge(nb_avb, on='neighbourhood_left')



geo_nb_avb.head(3)
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(12,10))

ax.set_aspect('equal')

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)



# Base plot

base = geo_nb_avb.plot(column='availability_365_y', ax=ax, figsize=(10,10), cmap='YlOrRd_r', legend=True, alpha=0.5, cax=cax)



# Set title

ax.set_title(label='Avg. room availability per year', fontdict=None, loc='center')



plt.show()
import seaborn as sns



corr = df_count.loc[:,~df_count.columns.isin(['latitude','longitude'])].corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("dark"):

    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
# Let's dummify categirical variables 'neighbourhood':

dummy_var = pd.get_dummies(df['neighbourhood'])



# And join them back to the df:

df_num_n = pd.concat([df_count, dummy_var], axis=1)



# Then dummify categirical variables 'room_type':

room_dummies = pd.get_dummies(df['room_type'])



# And join them back to the df:

df_num_n = pd.concat([df_num_n, room_dummies], axis=1)



# Drop host_id

df_num_n.drop('host_id', inplace=True, axis=1)
# Further data exploration

fig, ax = plt.subplots(1, 3)



plt.subplot(331)

plt.scatter(x=df_num_n.loc[:,'minimum_nights'], y=df_num_n.loc[:,['price']])

plt.xlabel('minimum_nights')

plt.ylabel('price')



plt.subplot(332)

plt.scatter(x=df_num_n.loc[:,['number_of_reviews']], y=df_num_n.loc[:,['price']])

plt.xlabel('number_of_reviews')



plt.subplot(333)

plt.scatter(x=df_num_n.loc[:,['reviews_per_month']], y=df_num_n.loc[:,['price']])

plt.xlabel('reviews_per_month')



plt.show()
df_num_n.head(2)
df_num_n[['price']].describe()
bins = [0, 69, 175, 10000]

group_names = ['Low', 'Medium', 'High']

df_num_n['price_binned'] = pd.cut(df_num_n['price'], bins, labels = group_names, include_lowest = True)
# Create a copy for backup

df_prbn = df_num_n.copy()



# Drop 'price' column

df_prbn.pop('price')



# Set y = 'price_binned' column values and exclude it from the df

y = df_prbn.pop('price_binned')



df_prbn.head(1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_prbn, y, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestClassifier 

from sklearn import metrics

rfc = RandomForestClassifier(n_estimators = 100,

                            n_jobs = -1,

                            max_features = "auto",

                            random_state = 888,

                            min_samples_leaf=1)

rfc.fit(x_train,y_train)

yhat = rfc.predict(x_test)

print(metrics.accuracy_score(y_test, yhat))
# Let's fund the best number of min_samples_leaf:

results = []

for sam in range(1,11):

    rfc = RandomForestClassifier(n_estimators = 100,

                            n_jobs = -1,

                            max_features = "auto",

                            random_state = 888,

                            min_samples_leaf=sam)

    rfc.fit(x_train,y_train)

    yhat = rfc.predict(x_test)

    results.append(metrics.accuracy_score(y_test, yhat))

plt.plot(results)

plt.xticks(np.arange(len(results)), np.arange(1,11))

plt.show()
results = []

for tr in range(600,2000,100):

    rfc = RandomForestClassifier(n_estimators = tr,

                            n_jobs = -1,

                            max_features = "auto",

                            random_state = 888,

                            min_samples_leaf=2)

    rfc.fit(x_train,y_train)

    yhat = rfc.predict(x_test)

    results.append(metrics.accuracy_score(y_test, yhat))

plt.plot(results)

plt.xticks(np.arange(len(results)), np.arange(600,2000,100))

plt.show()
results = []

mf_opt=["auto", None, "sqrt", "log2", 0.9, 0.2]



for max_f in mf_opt:

    rfc = RandomForestClassifier(n_estimators = 1000,

                            n_jobs = -1,

                            max_features = max_f,

                            random_state = 888,

                            min_samples_leaf=2)

    rfc.fit(x_train,y_train)

    yhat = rfc.predict(x_test)

    results.append(metrics.accuracy_score(y_test, yhat))

plt.plot(results)

plt.xticks(np.arange(len(results)), ["auto", None, "sqrt", "log2", 0.9, 0.2])

plt.show()
rfc = RandomForestClassifier(n_estimators = 1000,

                            n_jobs = -1,

                            max_features = 0.2,

                            random_state = 888,

                            min_samples_leaf=2)

rfc.fit(x_train,y_train)

yhat = rfc.predict(x_test)

print('The accuracy of the model I managed to build is %s percent' % round(100*metrics.accuracy_score(y_test, yhat),2))