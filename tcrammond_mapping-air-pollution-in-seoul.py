# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set_palette('dark')



import datetime



import os



import folium

from folium import * # This is the library for interactively visualizing data on the map
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

file_list = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
measurement_summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

measurement_item_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

measurement_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

measurement_station_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
print('Measurement Item info shape: {}'.format(measurement_item_info.shape))

measurement_item_info.head()
print('Measurement Station info shape: {}'.format(measurement_station_info.shape))

measurement_station_info.head()
print('Measurement Summary shape: {}'.format(measurement_summary.shape))

measurement_summary.head()
print('Measurement Info shape: {}'.format(measurement_info.shape))

measurement_info.head()
pollutants = measurement_item_info['Item name'].tolist()

print(pollutants)
measurement_station_info.set_index('Station code', inplace=True)
for p in pollutants:

    measurement_summary[measurement_summary[p] < 0] = 0
station_mean = measurement_summary.groupby(['Station code']).mean()

station_mean.drop(['Latitude', 'Longitude'], axis=1, inplace=True)

station_mean = station_mean.drop(station_mean.index[0])
pollutant_class = measurement_item_info.drop(['Item code', 'Unit of measurement'], axis=1).set_index('Item name')

pollutant_class.head(10)
def classifier(measurements, info, color=True):

    classified = pd.DataFrame(columns = measurements.columns)

    

    # classification to use

    if color:

        description = ['blue', 'green', 'yellow', 'red']

    else:

        description = ['Good', 'Normal', 'Bad', 'Very Bad']

    

    for i in measurements.index:

        for p in info.index:

            if measurements.loc[i, p] <= info.loc[p,'Good(Blue)']:

                classified.loc[i, p] = description[0]

            elif measurements.loc[i, p] <= info.loc[p, 'Normal(Green)']:

                classified.loc[i, p] = description[1]

            elif measurements.loc[i, p] <= info.loc[p, 'Bad(Yellow)']:

                classified.loc[i, p] = description[2]

            else:

                classified.loc[i, p] = description[3]

    return classified



means_classified = classifier(station_mean,pollutant_class)
measurement_summary['Measurement date'] = pd.to_datetime(measurement_summary['Measurement date'])

monthly_mean = measurement_summary.groupby(measurement_summary['Measurement date'].dt.month).mean()

monthly_mean.drop(['Station code', 'Latitude', 'Longitude'], axis=1, inplace=True)

monthly_mean.rename_axis('Month', inplace=True)

monthly_mean.head(12)
# chart each one of the means per month



fig, axs = plt.subplots(2,3, figsize=(12,8), tight_layout=True)



sns.barplot(monthly_mean.index, monthly_mean['SO2'], ax=axs[0,0]).set_title('SO2')

sns.barplot(monthly_mean.index, monthly_mean['NO2'], ax=axs[0,1]).set_title('NO2')

sns.barplot(monthly_mean.index, monthly_mean['CO'], ax=axs[0,2]).set_title('CO')

sns.barplot(monthly_mean.index, monthly_mean['O3'], ax=axs[1,0]).set_title('O3')

sns.barplot(monthly_mean.index, monthly_mean['PM10'], ax=axs[1,1]).set_title('PM10')

sns.barplot(monthly_mean.index, monthly_mean['PM2.5'], ax=axs[1,2]).set_title('PM2.5')



plt.show()
# This creates the map object

m = folium.Map(

    location=[37.541, 126.981], # center of where the map initializes

    tiles='Stamen Toner', # the style used for the map (defaults to OSM)

    zoom_start=12) # the initial zoom level



# Diplay the map

m
means_classified = classifier(station_mean,pollutant_class)
def pollutant_map(pollutant, measurements, station_info):

     

    # takes an input of a pollutant reference sheet, classified measurement data per station, and measurement

    # station information and outputs a Foilum Map with one layer for each pollutant type

    

    

    #initialize the folium map

    m = folium.Map(

    location=[37.541, 126.981], 

    tiles='Stamen Toner',

    zoom_start=11)



    for p in pollutants:

        feature_group = FeatureGroup(name=p, show=False)

        

        for i in means_classified.index:

            feature_group.add_child(Marker(station_info.loc[i, ['Latitude', 'Longitude']],

                         icon=folium.Icon(color=measurements.loc[i, p],

                                          icon='warning',

                                          prefix='fa')))

            m.add_child(feature_group)

            

    m.add_child(folium.map.LayerControl())

    m.save('pollutant_map.html')



    return m



pollutant_map(pollutant_class, means_classified, measurement_station_info)