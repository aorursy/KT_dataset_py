# Load libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import folium

from folium.plugins import HeatMap



# Import data

crimedata = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')



crime0 = crimedata.loc[crimedata['YEAR'].isin([2016,2017])]



crime = crime0.loc[crime0['UCR_PART'] == 'Part One']



#Remove unused columns

del crime['INCIDENT_NUMBER'] 

del crime['OFFENSE_CODE']

del crime['UCR_PART']

del crime['Location']



# Peek

crime.head()
crime[["OCCURRED_ON_DATE"]] = crime[["OCCURRED_ON_DATE"]].apply(pd.to_datetime)



# Convert OCCURED_ON_DATE to datetime





# Fill in nans in SHOOTING column

crime.SHOOTING.fillna('N', inplace=True)



# Convert DAY_OF_WEEK to an ordered category

crime.DAY_OF_WEEK = pd.Categorical(crime.DAY_OF_WEEK, 

              categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],

              ordered=True)



# Replace -1 values in Lat/Long with Nan

crime.Lat.replace(-1, None, inplace=True)

crime.Long.replace(-1, None, inplace=True)



# Rename columns to something easier to type (the all-caps are annoying!)

rename = {'OFFENSE_CODE_GROUP':'Group',

         'OFFENSE_DESCRIPTION':'Description',

         'DISTRICT':'District',

         'REPORTING_AREA':'Area',

         'SHOOTING':'Shooting',

         'OCCURRED_ON_DATE':'Date',

         'YEAR':'Year',

         'MONTH':'Month',

         'DAY_OF_WEEK':'Day',

         'HOUR':'Hour',

         'STREET':'Street'}

crime.rename(index=str, columns=rename, inplace=True)



# Check

crime.head()
print('There are '+str(crime.shape[0])+' incidents.')
# some data checks

crime.shape
# checking null values

crime.isnull().count()
sns.catplot(y='Group',

           kind='count',

            height=8, 

            aspect=1.5,

            order=crime.Group.value_counts().index,

           data=crime)
# Crimes by hour of the day

sns.catplot(x='Hour',

           kind='count',

            height=8.27, 

            aspect=3,

            color='red',

           data=crime)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('Hour', fontsize=40)

plt.ylabel('Count', fontsize=40)
array = ['Larceny']

larceny = crime.loc[crime['Group'].isin(array)]



array2 = ['Homicide']

homicide = crime.loc[crime['Group'].isin(array2)]
# Crimes by hour of the day

sns.catplot(x='Hour',

           kind='count',

            height=8.27, 

            aspect=3,

            color='red',

           data=larceny)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('Hour', fontsize=40)

plt.ylabel('Count', fontsize=40)
crime.groupby('Day').count()
# Crimes by day of the week

sns.catplot(x='Day',

           kind='count',

            height=10, 

            aspect=3,

           data=crime)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
larceny.groupby('Day').count()
# Crimes by day of the week

sns.catplot(x='Day',

           kind='count',

            height=10, 

            aspect=3,

           data=larceny)

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.catplot(x='Month', kind='count', height=8, aspect=3, color='gray', data=crime)

plt.xticks(np.arange(12), months, size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.catplot(x='Month', kind='count', height=8, aspect=3, color='gray', data=larceny)

plt.xticks(np.arange(12), months, size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sns.catplot(x='Month', kind='count', height=8, aspect=3, color='gray', data=homicide)

plt.xticks(np.arange(12), months, size=30)

plt.yticks(size=30)

plt.xlabel('')

plt.ylabel('Count', fontsize=40)
sns.scatterplot(x='Lat',

               y='Long',

                alpha=0.01,

               data=crime)
sns.scatterplot(x='Lat',

               y='Long',

                alpha=0.01,

               data=larceny)
sns.scatterplot(x='Lat',

               y='Long',

                hue='District',

                alpha=0.01,

               data=crime)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
sns.scatterplot(x='Lat',

               y='Long',

                hue='Group',

                alpha=0.01,

               data=crime)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
# Create basic Folium crime map

crime_heatmap = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "OpenStreetMap",

                      zoom_start = 11)



# Add data for heatmp 

data_heatmap = crime[crime.Year == 2017]

data_heatmap = crime[['Lat','Long']]

data_heatmap = crime.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_heatmap)



# Plot

crime_heatmap
# Create basic Folium crime map

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "OpenStreetMap",

                      zoom_start = 11)



# Add data for heatmp 

data_heatmap = larceny[larceny.Year == 2017]

data_heatmap = larceny[['Lat','Long']]

data_heatmap = larceny.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



#Plot

crime_map
# Create basic Folium crime map

crime_map = folium.Map(location=[42.3125,-71.0875], 

                       tiles = "OpenStreetMap",

                      zoom_start = 11)



# Add data for heatmp 

data_heatmap = homicide[homicide.Year == 2017]

data_heatmap = homicide[['Lat','Long']]

data_heatmap = homicide.dropna(axis=0, subset=['Lat','Long'])

data_heatmap = [[row['Lat'],row['Long']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)



# Plot

crime_map