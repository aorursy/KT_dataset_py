import pandas as pd

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap



df = pd.read_csv('../input/crime.csv')



# On use rows with geographical information for 2017

df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]



# Create a dataset of vehicle thefts in 2017

veh2017 =  df[(df['YEAR'] == 2017) & (df['TYPE'] == "Theft of Vehicle")]



# Create a map centered on Vancouver

map_van = folium.Map(location= [49.24, -123.11], zoom_start = 12)



# Create a list with lat and long values and add the list to a heat map, then show map

heat_data = [[row['Latitude'],row['Longitude']] for index, row in veh2017.iterrows()]

HeatMap(heat_data).add_to(map_van)



map_van
# Create a dataset of counts per hour 

veh_hour = veh2017[['TYPE','HOUR']].groupby(['HOUR'],as_index=False).count()



ax = veh_hour.plot(kind='bar', x='HOUR', title ="When are Vechical Thefts Happening", figsize=(15, 8), fontsize=12,legend=True,)

ax.set_xlabel("Hour of the Day", fontsize=12)

ax.set_ylabel("Thefts/Hour", fontsize=12)

plt.show()
map_van2 = folium.Map(location= [49.24, -123.11], zoom_start = 12)



# Create a dataset for 2017 vehicle thefts at 6pm (the worst time for thefts)

veh2017_16 =  veh2017[(veh2017['HOUR'] == 18)]



# Create a list with lat and long values and add the list to a heat map, then show map

heat_data = [[row['Latitude'],row['Longitude']] for index, row in veh2017_16.iterrows()]

HeatMap(heat_data).add_to(map_van2)



map_van2