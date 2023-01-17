# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.patches as patches
# Import data

survey_data = pd.read_csv("https://cocl.us/datascience_survey_data")

survey_data.head()
# Rename first column name

survey_data = survey_data.rename(columns = {'Unnamed: 0':'Topics'}, inplace = False)



#Set 'Topics' as index

survey_data.set_index("Topics", inplace=True)

survey_data
# Sort data by "Very Interested" column

survey_data = survey_data.sort_values(by='Very interested', ascending=False)



# Change data to percentages

survey_data_pct = survey_data/2233.00*100.00
# Visualize the percentage of the respondents' interest in the different data science topics surveyed



# Set color for each topic

colors_list = ['#5cb85c', '#5bc0de', '#d9534f']



# Create bar chart

ax = survey_data_pct.plot(kind='bar', 

                     figsize=(20, 8),

                     color=colors_list,

                     width=0.8,

                     rot=90,

)

plt.xlabel('', fontsize=14)

plt.ylabel('', fontsize=14)

plt.title("Percentage of Respondents' Interests in Data Science Areas", fontsize=16)

plt.xticks(fontsize = 14)

plt.legend(prop={"size":14})



# Add data label

for i in ax.patches:

    ax.annotate("%.2f" % i.get_height(), 

                (i.get_x() + i.get_width() / 2., 

                 i.get_height()), 

                ha='center', 

                va='center', 

                xytext=(0, 10), 

                textcoords='offset points',

               fontsize=14)



# Hide Y-axis

ax.axes.yaxis.set_visible(False)



# Remove left, top, right borders

ax.spines["left"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)
# Setup for geospatial charts



!conda install -c conda-forge folium=0.5.0 --yes

import folium



print('Folium installed and imported!')
# Import Data

sf_crime = pd.read_csv("https://cocl.us/sanfran_crime_dataset")



# Rename columns

sf_crime = sf_crime.rename(columns = {'PdDistrict':'District'}, inplace = False)

sf_crime
# Count total number of crimes by neighbourhood

crime_by_hood = sf_crime.groupby('District').IncidntNum.count()



# Convert to dataframe

df_hood = pd.DataFrame(crime_by_hood)



# Reset index

df_hood.reset_index(level=0, inplace=True)



# Rename column

df_hood = df_hood.rename(columns = {'IncidntNum':'Total_Crime'}, inplace = False)

df_hood
# download San Fransisco geojson file

!wget --quiet https://cocl.us/sanfran_geojson

    

print('GeoJSON file downloaded!')
# San Francisco latitude and longitude values

latitude = 37.77

longitude = -122.42



# create map and display it

sf_crime_map = folium.Map(location=[latitude, longitude], zoom_start=12)



# display the map of San Francisco

sf_crime_map
sanfran_geo = r'https://cocl.us/sanfran_geojson'



sf_crime_map.choropleth(geo_data=sanfran_geo,

                       data=df_hood,

                       columns=['District', 'Total_Crime'],

                       key_on='feature.properties.DISTRICT',

                       fill_color='YlOrRd', 

                       fill_opacity=0.7, 

                       line_opacity=0.2,

                       legend_name='Crime Rate in San Fransisco'

                      )



# Display map

sf_crime_map