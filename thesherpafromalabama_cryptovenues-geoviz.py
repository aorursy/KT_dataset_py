! pip install reverse_geocoder
# Run this to get data file path
import os
[print (i) for i in os.walk("../input")]
# Want to do some map plotting :)

import pandas as pd
import reverse_geocoder as rg
from pathlib import Path
import geopandas
import matplotlib.pyplot as plt
import requests
import io
import seaborn as sns
import numpy as np

df_path = Path(r'../input/blockchain-crypto-venues-jan-2018-may-2020/2LCountryCodes_post2018.csv')
df_path
venues_df = pd.read_csv(df_path)
venues_df.shape
venues_df.head()
response = requests.get('https://pkgstore.datahub.io/JohnSnowLabs/country-and-continent-codes-list/country-and-continent-codes-list-csv_csv/data/b7876b7f496677669644f3d1069d3121/country-and-continent-codes-list-csv_csv.csv')

file_object = io.StringIO(response.content.decode('utf-8'))

country_codes_df = pd.read_csv(file_object)
country_codes_df = country_codes_df.iloc[:,0:5]

# We have an issue: duplicate countries in different continents?? Let's drop these bad boys
country_codes_df[country_codes_df.Two_Letter_Country_Code == 'AZ']
# Get rid of duplicates
country_codes_df = country_codes_df.drop_duplicates('Two_Letter_Country_Code')
# merge this into our original df
venues_df = venues_df.merge(country_codes_df, right_on='Two_Letter_Country_Code', left_on='Country', how='left')
venues_df.drop(['Two_Letter_Country_Code'], axis = 1)
# venues_df.resample('M', on='Created_On').sum()
venues_df['M-Y'] = pd.to_datetime(venues_df[['Year', 'Month']].assign(DAY=1)).dt.date
venues_df['M-Y']
month_data = venues_df['M-Y'].value_counts()
month_data
# Plot the responses for different events and regions
fig, ax = plt.subplots(figsize=(15,15))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax = sns.countplot(x='M-Y', data=venues_df.sort_values(by="M-Y"), hue = 'Year') 
venues_df['M_Y']
# Need to convert to geopandas df
gdf = geopandas.GeoDataFrame(venues_df, geometry=geopandas.points_from_xy(venues_df.Long, venues_df.Lat))
print(gdf.head())
# Let's get a look at the world!

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

# We restrict to South America.
ax = world[world.continent != 'Antarctica'].plot(
    color='white', edgecolor='black', figsize = (25,25))

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color='red', markersize = 7)

plt.show()
# get the values of each country(code)
cc_srs = venues_df['Three_Letter_Country_Code'].value_counts()
cc_df = cc_srs.to_frame().reset_index()
cc_df.columns = ['3CC','Num_New_Vendors']

sns.barplot(data = cc_df, x = '3CC', y = 'Num_New_Vendors')
# Convert value counts series to geopandas
cc_gdf = geopandas.GeoDataFrame(cc_df)

# Get world dataset and add above data to it
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world.head()
world = world.merge(cc_gdf, left_on = 'iso_a3', right_on = '3CC', how='left')
world
world.plot(column = 'Num_New_Vendors', cmap='RdPu', figsize = (25,25))
world['Num_New_Vendors'] = np.log(world['Num_New_Vendors'])

world.plot(column = 'Num_New_Vendors', cmap='RdPu', figsize = (25,25))
base = world.plot(column = 'Num_New_Vendors', cmap='RdPu', figsize = (25,25))

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=base, color='lightgreen', markersize = 3)
eur_countries = world[world.continent == 'Europe'][world.name != 'Russia']
eur_countries

eur_gdf = gdf[gdf.Continent_Name == 'Europe'][gdf.Country_Name != 'Russian Federation']
eur_gdf['Country_Name'].value_counts()

world[world.continent == 'Europe']
dir(eur_countries[eur_countries['name'] == 'France'].geometry)
eur_countries[eur_countries['name'] == 'France'].geometry
base = eur_countries.plot(column = 'Num_New_Vendors', cmap='RdPu', figsize = (25,25))

# We can now plot our ``GeoDataFrame``.
eur_gdf.plot(ax=base, color='lightgreen', markersize = 7)
