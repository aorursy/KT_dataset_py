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
crime = pd.read_csv('../input/Crimes_-_2019.csv')

print ('File Read!')
crime.columns
crime.head()
crime.tail()
crime['Date'].min()
crime['Date'].max()
crime.shape
crime.dtypes
crime2=crime.dropna()

crime2
crime['Arrest'].value_counts()
crime2.shape
crime2['Primary Type'].value_counts() 
arrest=crime2[['Arrest','IUCR','Primary Type']]

arrest
caught = arrest[arrest.Arrest != False]

caught
caught['IUCR'].value_counts()
import folium
Latitude = -87.6298

Longitude = 41.8781
CHIcrime_map = folium.Map(location=[Longitude,Latitude], zoom_start=11)

CHIcrime_map
incidents = folium.map.FeatureGroup()
for lat, lng, in zip(crime2.Latitude, crime2.Longitude):

    incidents.add_child(

        folium.features.Marker(

            [lat, lng],

            radius=5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6

        )

    )



# add incidents to map

CHIcrime_map.add_child(incidents)