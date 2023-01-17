# Importing libraries



import numpy as np # data processing

import matplotlib.pyplot as plt # Visualization

import seaborn as sns # Visualization

import folium # Visualization Map

from folium.plugins import HeatMap # Visualization Map

from mpl_toolkits.basemap import Basemap # Visualization Map

# Importing Data

import pandas as pd

house = pd.read_csv("../input/california-housing-prices/housing.csv")
# Basic information about dataset

print(house.describe())
# Find rows and columns from this dataset

# Display column info

print("The number of rows and columns: " + str(house.shape))

print('\nThe columns are: \n')

[print(i,end='.\t\n') for i in house.columns]
# Display the first five rows in the dataset

print(house.head())
# Display the last five rows in the dataset

print(house.tail())
# Show all information about this dataset, which are columns, data type, memory usage and so on.

print(house.info())
# Check any NULL value in the dataset

print(house.isnull().sum())
# Display NULL data value using heatmap

plt.figure(figsize=(15,8))

plt.title('Missing data')

plt.ylabel("Count")

house.isnull().sum().plot(kind= 'bar' )
# Fix NUll value

median = house["total_bedrooms"].median()

house["total_bedrooms"].fillna(median, inplace=True)

print(house.isnull().sum())
# Count dataset

print(house.count())
# histogram of each columns data

house.hist(bins=80, figsize=(15, 15))
# Pie chart to show ocean proximity value

plt.figure(figsize=(8, 8))

plt.title("Pie chart on ocean proximity value")

house['ocean_proximity'].value_counts().plot(kind = 'pie',colormap = 'jet')





# amount on ocean_proximity categories

# X axis： house amount

# Y axis: ocean proximity

plt.figure(figsize=(12, 8))

sns.countplot(data=house, x="ocean_proximity")

plt.xlabel("Ocean Proximity")

plt.ylabel("House Amount")

plt.title("Number of houses on ocean proximity categories")
# Density of Median House Value on Ocean Proximity

plt.figure(figsize=(12, 8))

sns.stripplot(data=house, x="ocean_proximity", y="median_house_value",

              jitter=0.3)

plt.xlabel("Ocean Proximity")

plt.ylabel("Median House Value")

plt.title("Density of Median House Value on Ocean Proximity")
# house value on ocean_proximity categories

# X axis: Ocean Proximity

# Y axis: Median House Value

plt.figure(figsize=(12, 8))

sns.boxplot(data=house, x="ocean_proximity", y="median_house_value",

            palette="viridis")

plt.xlabel("Ocean Proximity")

plt.ylabel("Median House Value")

plt.title("House value on Ocean Proximity Categories")
#heatmap using seaborn

#set the context for plotting 

sns.set(context="paper",font="monospace")

housing_corr_matrix = house.corr()

#set the matplotlib figure

fig, axe = plt.subplots(figsize=(12,8))

#Generate color palettes 

cmap = sns.diverging_palette(220,10,center = "light", as_cmap=True)

#draw the heatmap

plt.title("Correlation between features")

sns.heatmap(housing_corr_matrix,vmax=1,square =True, cmap=cmap,annot=True );



print('\nAs shown in the Heatmap there is a strong correlation between the following features:\n')

print('- households')

print('- total_bedrooms')

print('- total_rooms')

print('- population')



print('\n')



print('The number of bedrooms in a district is obviously correlated\nwith the number of rooms in the district, the same is true for the number of families\nand the total population living in a district, finally number  of rooms is correlated\nwith the people\n')

#Finding Outliers

plt.figure(figsize=(15,5))

sns.boxplot(x=house['housing_median_age'])

plt.figure()

plt.figure(figsize=(15,5))

sns.boxplot(x=house['median_house_value'])
# histogram to show outliers on "median_house_value" columns

plt.figure(figsize=(12,8))

plt

plt.xlabel("Median House Value")

plt.ylabel("House Amount")

house['median_house_value'].hist(bins=100)
# Reomve outliers

house = house.loc[house['median_house_value']<500001,:]

plt.figure(figsize=(12,8))

plt.xlabel("Median House Value")

plt.ylabel("House Amount")

house['median_house_value'].hist(bins=100)
# Find the location on map

m = Basemap(projection='mill',llcrnrlat=25,urcrnrlat=49.5,\

            llcrnrlon=-140,urcrnrlon=-50,resolution='l')



plt.figure(figsize=(25,17))

m.drawcountries() 

m.drawstates()  

m.drawcoastlines()

x,y = m(-119.4179,36.7783)

m.plot(x, y, 'ro', markersize=20, alpha=.8) 

m.bluemarble() 

m.drawmapboundary(color = '#FFFFFF')
# Join Geographical Chart and histogram to show population density

plt.figure(figsize=(15,10))

sns.jointplot(x=house.latitude.values,y=house.longitude.values,size=10)

plt.ylabel("longitude")

plt.xlabel("latitude")
# Geographical Chart shows median house value

house.plot(kind="scatter", x='longitude', y='latitude', figsize=(15, 10),

           c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, )

plt.title("Geographical chart Shows Median House Value")
# Live Heatmap to show California state

map = folium.Map(location=[36.7783,-119.4179],

                    zoom_start = 6, min_zoom=5) 



df = house[['latitude', 'longitude']]

data = [[row['latitude'],row['longitude']] for index, row in df.iterrows()]

HeatMap(data, radius=10).add_to(map)

map
