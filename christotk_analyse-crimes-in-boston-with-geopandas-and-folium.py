import pandas as pd # data processing

import geopandas as gpd # geospatial data processing

import numpy as np # linear algebra

import folium # mapping

from folium.plugins import HeatMap

import seaborn as sns # visualization

import matplotlib.pyplot as plt # visualization

%matplotlib inline



# read crimes file

crimes = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding = 'latin')



# read Police Districts shapefile with geopandas

gdf = gpd.read_file('../input/boston-police-districts/police_districts/Police_Districts.shp')
gdf
crimes.head()
crimes.shape
crimes.describe()
crimes.isnull().sum()
gdf.plot()

plt.tight_layout()
gdf['point'] = gdf.representative_point() # this is a point guaranteed to be within each polygon



# label_points - a GeoDataFrame used for labeling

label_points = gdf.copy()

label_points.set_geometry('point', inplace = True)



# plot districts

ax = gdf.plot(color = 'whitesmoke', figsize = (12,8), edgecolor = 'black', linewidth = 0.3)



def add_label():

    # add label for each polygon

    for x, y, label in zip(label_points.geometry.x, label_points.geometry.y, label_points['DISTRICT']):

        plt.text(x, y, label, fontsize = 10, fontweight = 'bold')



add_label()

plt.title('Boston police districts', fontsize = 16)

plt.tight_layout()
most_common_crimes = pd.DataFrame({'Count': crimes.OFFENSE_CODE_GROUP.value_counts().sort_values(ascending = False).head(10)}) # top 10 most common crimes

most_common_crimes
plt.figure(figsize = (20,12))

sns.barplot(x = most_common_crimes.index, y = 'Count', data = most_common_crimes)

plt.yticks((np.arange(5000, most_common_crimes['Count'].max(), 5000)))

plt.ylabel(None)

plt.tick_params(labelsize = 12)

plt.xlabel('\n Most common crime types', fontsize = 12)

plt.title('Top 10 crimes in Boston', fontsize = 18)

plt.tight_layout()
location_of_most_common_crimes = crimes[crimes.OFFENSE_CODE_GROUP.isin(most_common_crimes.index)].loc[:, ['Lat', 'Long']].dropna()



my_map=folium.Map(location = [42.320,-71.05], #Initiate map on Boston city

                  zoom_start = 11,

                  min_zoom = 11

)



HeatMap(data=location_of_most_common_crimes.sample(10000), radius=16).add_to(my_map)



my_map
districts = pd.DataFrame({'Count': crimes.DISTRICT.value_counts().sort_values(ascending = False)})

districts
plt.figure(figsize = (12,8))

sns.barplot(x = districts.index, y = 'Count', data = districts, palette = 'Reds_r')

sns.lineplot(x = districts.index, y = districts['Count'].mean(), data = districts, label = 'mean', color = 'black') # plot mean value

plt.title('Crimes per district in Boston', fontsize = 16)

plt.ylabel(None)

plt.xlabel('\nDISTRICT')

plt.yticks(np.arange(10000, 55000, 10000))

plt.tick_params(labelsize = 12)

plt.tight_layout()
gdf['crimes'] = gdf.DISTRICT.map(districts['Count']) # use map function to match each district with its corresponding value

ax = gdf.plot(column = gdf.crimes, cmap = 'Reds', legend = True, edgecolor = 'black', linewidth = 0.3, figsize = (12,8))

add_label()

plt.title('Crimes per district in Boston', fontsize = 16)

plt.tight_layout()
crimes_per_hour = pd.DataFrame({'Count': crimes['HOUR'].value_counts().sort_index()})

crimes_per_hour
plt.figure(figsize = (12,8))

sns.barplot(x = crimes_per_hour.index, y = crimes_per_hour['Count'], data = crimes_per_hour, color = '#7AD7F0')

plt.ylabel(None)

plt.xlabel(None)

plt.yticks(np.arange(2500, 22000, 2500))

plt.tick_params(labelsize = 12)

plt.title('Boston crimes per hour', fontsize = 16)

plt.tight_layout()
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

crimes_per_day = pd.DataFrame({'Count': crimes['DAY_OF_WEEK'].value_counts().loc[labels]})

crimes_per_day
plt.figure(figsize = (12,8))

sns.barplot(x = crimes_per_day.index, y = 'Count', data = crimes_per_day)

plt.ylabel(None)

plt.xlabel(None)

plt.yticks(np.arange(10000, 55000, 10000))

plt.tick_params(labelsize = 12)

plt.title('Boston crimes per day', fontsize = 16)

plt.tight_layout()
print(crimes.OCCURRED_ON_DATE.min())

print(crimes.OCCURRED_ON_DATE.max())
crimes_2016_2017 = crimes[crimes['YEAR'].isin(['2016', '2017'])]

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

crimes_per_month = pd.DataFrame({'Count': crimes_2016_2017['MONTH'].value_counts().sort_index().values}, index = months)

crimes_per_month
plt.figure(figsize = (14, 8))

sns.barplot(x = crimes_per_month.index, y = 'Count', data = crimes_per_month, palette = 'tab10')

plt.ylabel(None)

plt.xlabel(None)

plt.yticks(np.arange(2500, 20000, 2500))

plt.tick_params(labelsize = 12)

plt.title('Boston crimes per month 2016 - 2017', fontsize = 16)

plt.tight_layout()
crimes_per_year = pd.DataFrame({'Count': crimes_2016_2017['YEAR'].value_counts().sort_index()})

crimes_per_year
plt.figure(figsize = (12, 8))

sns.barplot(x = crimes_per_year.index, y = 'Count', data = crimes_per_year)

plt.ylabel(None)

plt.tick_params(labelsize = 12)

plt.yticks(np.arange(20000, 120000, 20000))

plt.title('Boston crimes per year 2016 - 2017', fontsize = 16)

plt.tight_layout()
crimes_per_year['population'] = [678430, 685094] # Boston population for 2016 and 2017
(crimes_per_year.loc[2017].Count-crimes_per_year.loc[2016].Count)/crimes_per_year.loc[2016].Count
(crimes_per_year.loc[2017].population-crimes_per_year.loc[2016].population)/crimes_per_year.loc[2016].population