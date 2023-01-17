import pandas as pd



raw_data = pd.read_csv('/kaggle/input/singapore-train-station-coordinates/mrt_lrt_data.csv')



print('Total stations amount:', len(raw_data))

print(raw_data.describe())

raw_data.head()
print('NaN values amount:')

print(raw_data.isna().sum())
mrt_stations = raw_data.loc[raw_data['type'] == 'MRT']

lrt_stations = raw_data.loc[raw_data['type'] == 'LRT']

print('MRT stations amount:', len(mrt_stations))

print('LRT stations amount:', len(lrt_stations))
import seaborn as sns

import folium

from folium.plugins import HeatMap

import matplotlib.pyplot as plt



singapore_map = folium.Map(location=[1.3521, 103.8198],

                           zoom_start = 12)



mrt_coordinates = mrt_stations[['lat', 'lng']]

lrt_coordinates = lrt_stations[['lat', 'lng']]



mrt_heatmap = HeatMap(mrt_coordinates, radius=20, gradient={0: 'white', 1:'red'})

lrt_heatmap = HeatMap(lrt_coordinates, radius=20, gradient={0: 'white', 1:'blue'})



mrt_heatmap.add_to(singapore_map)

lrt_heatmap.add_to(singapore_map)

singapore_map