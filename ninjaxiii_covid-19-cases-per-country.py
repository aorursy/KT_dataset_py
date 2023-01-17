import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Install required libraries

!pip install folium

import requests

import folium
# Create a dataframe from the dataset



countryData = pd.read_csv('../input/covid19-cases/covid19_preprocessed.csv')



countryData.head(10)
# Import the worldcountries file

worldMapFile = '../input/world-countries/world-countries.json'



# Create an empty world map using folium

worldMap = folium.Map(location=[40, 0], zoom_start=1.5)
# Plotting the data on the map with a custom scale



worldMap.choropleth(

    geo_data = worldMapFile, 

    data = countryData, 

    columns=['Country', 'TotalCases'],

    key_on='feature.properties.name',

    fill_color='YlOrRd',

    threshold_scale=[0, 1000, 10000, 30000, 75000, 100000],

    fill_opacity=0.6, 

    line_opacity=0.3,

    legend_name='COVID19 Total Cases'

)



worldMap