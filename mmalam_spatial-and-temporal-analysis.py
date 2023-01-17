import pandas as pd

import numpy as np

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

from shapely.geometry import Point, Polygon

import matplotlib.gridspec as gridspec
data = pd.read_csv('../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv')

data.head()
import geopandas as gpd

from shapely.geometry import Point, Polygon

from mpl_toolkits.axes_grid1 import make_axes_locatable
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

usa = world[world['name'] == 'United States of America']

fig,ax = plt.subplots(figsize = (10,10))

usa.plot(ax = ax, color='blue')
border_crossing = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data2.csv')

border_crossing.head()
crs = {'init': 'epsg:4326'} #Coordinate Reference System

geometry = [Point(xy) for xy in zip( border_crossing["Longitude"], border_crossing["Latitude"])]

ports = gpd.GeoDataFrame(border_crossing, crs = crs, geometry = geometry)

ports.head()
fig,ax = plt.subplots(figsize = (20,50))

title = plt.title('Ports at USA Borders', fontsize=20)

title.set_position([0.5, 1.05])

usa.plot(ax = ax, color='grey', edgecolor='black',linewidth=1, alpha=0.1)

ports.plot(marker='o', color='green', markersize=5, ax=ax)
states = gpd.read_file('../input/us-border-crossing-temporal-and-spatial-analysis/states.shp')

alaska = gpd.read_file('../input/us-border-crossing-temporal-and-spatial-analysis/alaska.shp')
#Number of Crossing by State



#SELECT [State] AS [STATE_NAME], COUNT(*) AS [Crossing]

#FROM [BC1]

#GROUP BY [State]

#ORDER BY [State]

state_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data3.csv')
state_count_gp = states.merge(state_count, on='STATE_NAME', how='left')

state_count_gp["Crossing"].fillna(0, inplace=True)

state_count = state_count.rename(columns={'STATE_NAME':'NAME'})

alaska_count_gp = alaska.merge(state_count, on='NAME', how='left')
f,ax=plt.subplots(1,2,figsize=(25,5))



title = plt.title('Number of Crossings by State', fontsize=20)

title.set_position([-0.2, 1.25])



alaska_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=60000,vmin=0)

state_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=60000,vmin=0)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
# Prepare Data: Adding types of veicle based on Measure column



# SELECT [STATE_NAME], Vehicle_Type, COUNT(*)

# FROM

# (

# SELECT [State] AS [STATE_NAME], 

# CASE 

# WHEN Measure = 'Bus Passengers' THEN 'Heavy Vehicle'

# WHEN Measure = 'Buses' THEN 'Heavy Vehicle'

# WHEN Measure = 'Pedestrians' THEN 'Pedestrians'

# WHEN Measure = 'Personal Vehicle Passengers' THEN 'Light Vehicle'

# WHEN Measure = 'Personal Vehicles' THEN 'Light Vehicle'

# WHEN Measure = 'Rail Containers Empty' THEN 'Heavy Vehicle'

# WHEN Measure = 'Rail Containers Full' THEN 'Heavy Vehicle'

# WHEN Measure = 'Train Passengers' THEN 'Train'

# WHEN Measure = 'Trains' THEN 'Train'

# WHEN Measure = 'Truck Containers Empty' THEN 'Heavy Vehicle'

# WHEN Measure = 'Truck Containers Full' THEN 'Heavy Vehicle'

# WHEN Measure = 'Trucks' THEN 'Heavy Vehicle'

# ELSE '' END AS [Vehicle_Type]

# FROM [BC1]

# ) AS B1

# GROUP BY [STATE_NAME], Vehicle_Type

# ORDER BY [STATE_NAME]



state_Heavy_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data_Heavy.csv')

state_Light_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data_Light.csv')

state_Train_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data_Train.csv')

state_Pedestrian_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data_Pedestrian.csv')



state_Heavy_count_gp = states.merge(state_Heavy_count, on='STATE_NAME', how='left')

state_Light_count_gp = states.merge(state_Light_count, on='STATE_NAME', how='left')

state_Train_count_gp = states.merge(state_Train_count, on='STATE_NAME', how='left')

state_Pedestrian_count_gp = states.merge(state_Pedestrian_count, on='STATE_NAME', how='left')



state_Heavy_count["Crossing"].fillna(0, inplace=True)

state_Light_count["Crossing"].fillna(0, inplace=True)

state_Train_count["Crossing"].fillna(0, inplace=True)

state_Pedestrian_count["Crossing"].fillna(0, inplace=True)



state_Heavy_count = state_Heavy_count.rename(columns={'STATE_NAME':'NAME'})

state_Light_count = state_Light_count.rename(columns={'STATE_NAME':'NAME'})

state_Train_count = state_Train_count.rename(columns={'STATE_NAME':'NAME'})

state_Pedestrian_count = state_Pedestrian_count.rename(columns={'STATE_NAME':'NAME'})



alaska_Heavy_count_gp = alaska.merge(state_Heavy_count, on='NAME', how='left')

alaska_Heavy_count_gp["Crossing"].fillna(0, inplace=True)

alaska_Light_count_gp = alaska.merge(state_Light_count, on='NAME', how='left')

alaska_Light_count_gp["Crossing"].fillna(0, inplace=True)

alaska_Train_count_gp = alaska.merge(state_Train_count, on='NAME', how='left')

alaska_Train_count_gp["Crossing"].fillna(0, inplace=True)

alaska_Pedestrian_count_gp = alaska.merge(state_Pedestrian_count, on='NAME', how='left')

alaska_Pedestrian_count_gp["Crossing"].fillna(0, inplace=True)
f,ax=plt.subplots(1,2,figsize=(25,5))



title = plt.title('Heavy Vehicles Crossings by State', fontsize=20)

title.set_position([-0.2, 1.25])



alaska_Heavy_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=35000,vmin=0)

state_Heavy_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=35000,vmin=0)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
f,ax=plt.subplots(1,2,figsize=(25,5))



title = plt.title('Light Vehicles Crossings by State', fontsize=20)

title.set_position([-0.2, 1.25])



alaska_Light_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=11000,vmin=0)

state_Light_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=11000,vmin=0)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
f,ax=plt.subplots(1,2,figsize=(25,5))



title = plt.title('Train Crossings by State', fontsize=20)

title.set_position([-0.2, 1.25])



alaska_Train_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=11000,vmin=0)

state_Train_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=11000,vmin=0)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
f,ax=plt.subplots(1,2,figsize=(25,5))



title = plt.title('Pedestrians Crossings by State', fontsize=20)

title.set_position([-0.2, 1.25])



alaska_Pedestrian_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[0]

                 , vmax=5000,vmin=0)

state_Pedestrian_count_gp.plot( column='Crossing', 

                  cmap='OrRd', legend=True, ax=ax[1]

                 , vmax=5000,vmin=0)

ax[0].axes.get_xaxis().set_visible(False)

ax[0].axes.get_yaxis().set_visible(False)

ax[1].axes.get_xaxis().set_visible(False)

ax[1].axes.get_yaxis().set_visible(False)
#SELECT *, MONTH([Date]) AS [Month], YEAR([Date]) AS [Year]

#FROM [BC1]

#ORDER BY [State], MONTH([Date])





data_ym = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data7.csv')

data_ym.head()
f,ax=plt.subplots(1,1,figsize=(20,10))

title = plt.title('Crossings by Year', fontsize=20)

title.set_position([0.5, 1.05])



sns.countplot('Year',data=data_ym,ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of Crossings')

g = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='center', fontsize=15)

plt.show()
f,ax=plt.subplots(1,1,figsize=(20,10))

title = plt.title('Crossings by Month', fontsize=20)

title.set_position([0.5, 1.05])



sns.countplot('Month',data=data_ym,ax=ax)

ax.set_xlabel('Month')

ax.set_ylabel('Number of Crossings')

x_label_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

g = ax.set_xticklabels(x_label_list, rotation=30, horizontalalignment='right', fontsize=15)

plt.show()
#SELECT MONTH([Date]) AS [Month], YEAR([Date]) AS [Year], COUNT(*) AS [Crossing]

#FROM [BC1]

#GROUP BY MONTH([Date]),YEAR([Date])





data_ym_count = pd.read_csv('../input/us-border-crossing-temporal-and-spatial-analysis/Border_Crossing_Entry_Data8.csv')
heatmap_data = pd.pivot_table(data_ym_count, values='Crossing', 

                     index=['Month'], 

                     columns='Year')



f,ax=plt.subplots(1,1,figsize=(20,7))

title = plt.title('Crossings by Month', fontsize=20)

title.set_position([0.5, 1.05])





g = sns.heatmap(heatmap_data, cmap="OrRd",linewidths=.5, linecolor='black', cbar_kws={'label': 'Crossings'}, ax=ax)

y_label_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

y = ax.set_yticklabels(y_label_list, rotation=0, horizontalalignment='right')

plt.show()