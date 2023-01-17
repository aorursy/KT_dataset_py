# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import geopandas as gpd

from shapely.geometry import LineString

from geopandas.tools import geocode

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, FastMarkerCluster

from folium import plugins

import math

import webbrowser

from IPython.display import HTML

import matplotlib.pyplot as plt

from pandasql import sqldf

import plotly.express as px



#turn off settingwithcopywarning off

pd.options.mode.chained_assignment = None



geo_code = True
#Have a look at initial dataset

US_covid_data_base_file = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv")

#Restrict dates to be before 30th April for performance reasons

US_covid_data_date_restricted = US_covid_data_base_file.loc[US_covid_data_base_file['date']<'2020-04-30']

print("Shape of initial dataset: " + str(US_covid_data_date_restricted.shape))

df1 = US_covid_data_date_restricted

df1
#Use one location as a test example

US_covid_data_cumulative_or_new_test = df1.loc[(df1['county']=='King')& (df1['state']=='Washington')]

plt.figure()



#set x and y variables

x = US_covid_data_cumulative_or_new_test['date']

y = US_covid_data_cumulative_or_new_test['cases']



#setting x ticks to be the start and end date so that the x-axis isn't messy

x_ticks = [x.min(),x.max()]



#plot variables to inspect if 'cases' column is cumulative cases or new cases per day

plt.plot(x, y)

plt.ylabel('Cases')

plt.xlabel('Date')

plt.xticks(x_ticks,rotation=70)

plt.title('King, Washington Cases')

plt.show
#Conditional on the rows where we have the max date, inspect the count of all the distinct county names

inspect_county_values = df1.loc[df1['date'] == df1['date'].max()]

inspect_county_values['county'].value_counts()
#Inspect the initial dataset for counties that dont have fips

#This is important for later when mapping with a separate shapefile that contains a list of GEOID's (fips)

#When inspecting eliminate counties = 'unknown' since you can't reconcile that to a county location

US_covid_nulls = df1.loc[(df1['county'] != 'Unknown')& (df1['date'] == df1['date'].max())]



US_covid_nulls = US_covid_nulls[US_covid_nulls.isnull().any(axis=1)].county.value_counts()



print("Counties which can be manually accounted for when mapping using shapefile: \n" + str(US_covid_nulls))

#Since cumulative cases, use just the max date so that you have the total cases to date

US_covid_data = df1.loc[df1.date == df1.date.max()]

print("Rows as of max date: \n\n" + str(US_covid_data['date'].value_counts()))
# Because it's possible the name of a county may exist in more than one state, concat the county name with the state name so that it is unique

US_covid_data['concat'] = df1['county']+str(', ')+df1['state']

#Inspect to see if there are any duplicates - there shouldnt be any 

US_covid_data['concat'].value_counts(ascending=False)

#Ascending = False so if first row equals 1 then every value in concat column is unique
#Geocode the concat column

if geo_code:

    def my_geocoder(row):

            try:

                point = geocode(row, provider='nominatim').geometry.iloc[0]

                return pd.Series({'Latitude': point.y, 'Longitude': point.x, 'geometry': point})

            except:

                return None



    US_covid_data[['Latitude', 'Longitude', 'geometry']] = US_covid_data.apply(lambda x: my_geocoder(x['concat']), axis=1)

    

    US_covid_data.to_csv('US_covid_data_maxdate_geocoded.csv', index=False)

else:

    US_covid_data = pd.read_csv('US_covid_data_maxdate_geocoded.csv')
#Fill in null fips value with arbitray number 

blank_fips_counties = ['New York City', 'Kansas City']



for i in blank_fips_counties:

    US_covid_data.loc[US_covid_data['county'] == i,'fips'] = 1





#Drop any rows where county = 'Unknown' through use of dropna() since fips value for every 'unknown' county is null. Note that any other locations that couldnt be geocoded will also be dropped 

US_covid = US_covid_data.dropna()

print("Percentage of rows that could be geocoded:\n"+str((US_covid.shape[0]/US_covid_data.shape[0])*100))
#In order to map, each row needs to be replicated based on the value in the 'cases' column

#For example if row x has a cases count of 635 then row x needs to appear 635 times

US_covid = US_covid.loc[US_covid.index.repeat(US_covid['cases'])]

print("Shape of dataset to be mapped: " +str(US_covid.shape))
#Create base map

map1 = folium.Map(location=[40, -95], zoom_start=4)



#Marker Cluster

map1.add_child(FastMarkerCluster(US_covid[['Latitude', 'Longitude']].values.tolist()))

#Heat Map

HeatMap(data=US_covid[['Latitude', 'Longitude']], radius=16.5, blur =16.5).add_to(map1)



map1.save('plot_data.html')   

HTML('<iframe src=plot_data.html width=800 height=600></iframe>')

        
#Read the file

us_counties_shapefile_base = gpd.read_file("../input/us-counties-geocoded/tl_2017_us_county.shp")

df2 = us_counties_shapefile_base

df2.head()
us_counties_dataframe = pd.DataFrame(df2[['NAME','GEOID', 'INTPTLAT', 'INTPTLON']])

us_counties_dataframe.to_csv('geocodes.csv', index = False)

us_counties_dataframe['GEOID'] = us_counties_dataframe['GEOID'].astype('float64')

#check df2 Kansas City, New York City and Joplin to see if you can manually account for these two when merging df1 and df2

for i in ["Kansas", "New York"]:

    check = us_counties_dataframe[us_counties_dataframe['NAME'].str.contains(i)]

    print("Check for "+str(i)+"\n" +str(check)+"\n")



#Can manually account for New York City as there is only one row that returns from df2

#Kansas City, Missouri - will have to get the coordinates from google
#Original Dataframe of Covid cases in USA

US_covid_data = df1

US_covid_data = US_covid_data.loc[US_covid_data.date == US_covid_data.date.max()]
print("NY City before update: \n" +str(US_covid_data.loc[US_covid_data['county'] == 'New York City'])+"\n\n\n")

US_covid_data.at[US_covid_data.loc[US_covid_data['county'] == 'New York City'].index[0],'fips'] = 36061.0

print("NY City after update: \n" +str(US_covid_data.loc[US_covid_data['county'] == 'New York City'])+"\n")

#Merge df1 and df2 together 

concat_result = US_covid_data.merge(us_counties_dataframe[['GEOID','INTPTLAT', 'INTPTLON']],left_on = 'fips', right_on = 'GEOID', how = 'left')

print("Rows in left df: "+str(US_covid_data.shape[0]))

print("Rows in joint df: "+str(concat_result.shape[0]))

concat_result

#Manually update Kansas City, Missouri with 39.0997°, -94.5786°

print("Kansas City before update: \n" +str(concat_result.loc[concat_result['county'] == 'Kansas City'])+"\n\n\n")

concat_result.at[concat_result.loc[concat_result['county'] == 'Kansas City'].index[0],'INTPTLAT'] = '+39.0997000'

concat_result.at[concat_result.loc[concat_result['county'] == 'Kansas City'].index[0],'INTPTLON'] = '-94.5786000'

#Also add in arbitrary numbers to fips and GEOID column so that this doesn't get dropped when you use dropna() later

concat_result.at[concat_result.loc[concat_result['county'] == 'Kansas City'].index[0],'fips'] = 1

concat_result.at[concat_result.loc[concat_result['county'] == 'Kansas City'].index[0],'GEOID'] = 1

print("Kansas City after update: \n" +str(concat_result.loc[concat_result['county'] == 'Kansas City'])+"\n")

#Remove the '+' from the latitude column so that it can be mapped 

concat_result['INTPTLAT'] = concat_result['INTPTLAT'].astype('str')

concat_result['INTPTLAT']

concat_result['INTPTLAT'] = concat_result['INTPTLAT'].str[1:]

concat_result['INTPTLAT']





#Remove rows with NaN's - this will be where county = Unknown

concat_result.dropna(how = 'any', inplace = True)



#Mapping the Covid Cases - required to duplicate rows based on value in 'cases' column

concat_result_cases = concat_result.loc[concat_result.index.repeat(concat_result['cases'])]

print("Shape of dataset to be mapped: " +str(concat_result_cases.shape))



#Convert latitude and longitude columns so that it's compatible with mapping

concat_result_cases['INTPTLAT'] = concat_result_cases['INTPTLAT'].astype('float64')

concat_result_cases['INTPTLON'] = concat_result_cases['INTPTLON'].astype('float64')

#Create base map

map2 = folium.Map(location=[40, -95], zoom_start=4)



#Marker Cluster

map2.add_child(FastMarkerCluster(concat_result_cases[['INTPTLAT', 'INTPTLON']].values.tolist()))

#Heat Map

HeatMap(data=concat_result_cases[['INTPTLAT', 'INTPTLON']], radius=16.5, blur = 16.5).add_to(map2)



map2.save('plot_data2.html')   

HTML('<iframe src=plot_data2.html width=800 height=600></iframe>')

        
#Mapping the Covid Deaths

concat_result_deaths = concat_result.loc[concat_result.index.repeat(concat_result['deaths'])]

concat_result_deaths = concat_result_deaths.loc[concat_result_deaths['deaths']!=0]

print("Shape of dataset to be mapped: " +str(concat_result_deaths.shape))



concat_result_deaths['INTPTLAT'] = concat_result_deaths['INTPTLAT'].astype('float64')

concat_result_deaths['INTPTLON'] = concat_result_deaths['INTPTLON'].astype('float64')



#Create base map

map3 = folium.Map(location=[40, -95], zoom_start=4)



#Marker Cluster

map3.add_child(FastMarkerCluster(concat_result_deaths[['INTPTLAT', 'INTPTLON']].values.tolist()))

#Heat Map

HeatMap(data=concat_result_deaths[['INTPTLAT', 'INTPTLON']], radius=16.5, blur = 16.5).add_to(map3)



map3.save('plot_data3.html')   

HTML('<iframe src=plot_data3.html width=800 height=600></iframe>')

        