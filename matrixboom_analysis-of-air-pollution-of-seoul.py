import numpy as np

import pandas as pd

import folium



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
measurement_summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

measurement_summary.head()
limit = 2

data = measurement_summary.sort_values('PM2.5',ascending=False).iloc[0:limit, :]

marks = folium.map.FeatureGroup()

for lat, lng, in zip(data.Latitude, data.Longitude):

    marks.add_child(

        folium.CircleMarker(

            [lat, lng],

            radius=7, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='red',

            fill_opacity=0.4

        )

    )

seoul_map = folium.Map(location=[37.572016, 127.005007], zoom_start=12)

seoul_map.add_child(marks)
data