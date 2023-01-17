# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import matplotlib.pyplot as plt #plotting library for data viz

import pandas as pd

import shapefile as shp

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cv = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

#cv_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_confirmed.csv")

#cv_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_deaths.csv")

#cv_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_2019_ncov_recovered.csv")

cv.head()
cv.columns

cv.dtypes
cv.groupby("Country")['Confirmed'].sum()
cv.groupby("Country")['Confirmed'].sum().plot(kind="bar")
cv.groupby("Country")['Deaths'].count()
cv.groupby("Country")['Deaths'].count().plot(kind="bar")
cv.groupby("Country")['Recovered'].count()
cv.groupby("Country")['Recovered'].count().plot(kind="bar")
cv.groupby("Country")['Confirmed','Deaths', 'Recovered'].sum().plot(kind="bar")
#cv.plot.pie(y=["Confirmed","Deaths", "Recovered"])
affected_countries = len(cv['Country'].value_counts())



cases = pd.DataFrame(cv.groupby('Country')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,affected_countries+1)



global_cases = cases[['Country','Confirmed']]

global_cases
world_coordinates = pd.read_csv("../input/world-coordinates/world_coordinates.csv")

world_coordinates.head()
world_data = pd.merge(world_coordinates,global_cases,on='Country')

world_data.head()
#This doesn't work

#wc = pd.read_csv('../input/hcde-cv-coordinates/world_coordinates.csv')
import folium
#cv_loc.drop(['Confirmed','Code'], 1)

cv_world_map = folium.Map(location=[cv_loc.latitude.mean(), cv_loc.longitude.mean()], tiles="OpenStreetMap", zoom_start = 2)





#  add Locations to map

for lat, lng, label in cv_loc:

    folium.Marker(location=[lat,lng],

            popup = label,

            icon = folium.Icon(color='green')

    ).add_to(marker_cluster)



map_world_NYC.add_child(marker_cluster)



#  display interactive map

cv_world_map

cv_world_map = MarkerCluster().add_to(map_world_NYC)



for lat, lng, label in zip(dfNY.Latitude, dfNY.Longitude, dfNY.Location):

    folium.Marker(location=[lat,lng],

            popup = label,

            icon = folium.Icon(color='green')

    ).add_to(marker_cluster)



map_world_NYC.add_child(marker_cluster)