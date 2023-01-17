import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap



data = pd.read_csv("../input/crime.csv")

# For this study only look at a few key columns ands remove rows with no data

df = data [['OFFENSE_ID','OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID','REPORTED_DATE','GEO_LON','GEO_LAT']]

df = df.dropna()
# Get a count of the top 20 crimes, based on the "OFFENSE_TYPE_ID"

crime_cnts = df[['OFFENSE_TYPE_ID','OFFENSE_ID']].groupby(['OFFENSE_TYPE_ID'],as_index=False).count().nlargest(20,['OFFENSE_ID'])



# Plot the most common crimes

ax = crime_cnts.plot(kind='bar', x='OFFENSE_TYPE_ID', title ="Overall Counts of the Type of Crime", figsize=(15, 8), fontsize=12,legend=False)

ax.set_xlabel("Types of Crime", fontsize=12)

ax.set_ylabel("Total Counts", fontsize=12)

plt.show()
import warnings

warnings.filterwarnings('ignore')



# Group all traffic crimes into a data set

traffic = df[df['OFFENSE_TYPE_ID'].str[:4] == 'traf']



# Create columns for month and hour

traffic['MONTH'] = pd.DatetimeIndex(traffic['REPORTED_DATE']).month

traffic['HOUR'] = pd.DatetimeIndex(traffic['REPORTED_DATE']).hour



# Which months are getting the most traffic crimes

traf_month = traffic[['OFFENSE_TYPE_ID','MONTH']].groupby(['MONTH'],as_index=False).count()

ax = traf_month.plot(kind='bar', x='MONTH', title ="Which Month is Getting the Most Traffic Crimes", figsize=(15, 8), fontsize=12,legend=False,)

ax.set_xlabel("Month", fontsize=12)

ax.set_ylabel("Crime/Month", fontsize=12)

plt.show()



# Which hours of the day are getting the most traffic crimes

traf_hour = traffic[['OFFENSE_TYPE_ID','HOUR']].groupby(['HOUR'],as_index=False).count()

ax = traf_hour.plot(kind='bar', x='HOUR', title ="Which Hour of the Day is Getting the Most Traffic Crimes", figsize=(15, 8), fontsize=12,legend=False,)

ax.set_xlabel("Hour", fontsize=12)

ax.set_ylabel("Crime/Hour", fontsize=12)

plt.show()
# Add mapping libraries and traffic summaries on a geographic map

import folium

from folium.plugins import HeatMap



map_den = folium.Map(location= [39.76,-105.02], zoom_start = 16)



# Get data from 15:00 to 18:00

den15_18 = traffic[(traffic['HOUR'] >= 15) & (traffic['HOUR'] <= 18)]



# Create a list with lat and long values and add the list to a heat map, then show map

heat_data = [[row['GEO_LAT'],row['GEO_LON']] for index, row in den15_18.iterrows()]

HeatMap(heat_data).add_to(map_den)



map_den