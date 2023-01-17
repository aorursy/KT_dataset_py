#Libraries Needed

import pandas as pd



#Settings to display pandas

pd.set_option('display.max_columns', None)



#Some basic set up

base_path = "/kaggle/input/uncover/"



#Using the Coders Against COVID as example

CAC_path = base_path + "coders_against_covid/"

CAC_file = CAC_path + "crowd-sourced-covid-19-testing-locations.csv"

CAC_df = pd.read_csv(CAC_file)

#Get one row as example

CAC_df.head(1)
import folium

from folium import plugins



#Preprocess

#Drop rows that do not have lat/lon

CAC_df = CAC_df[CAC_df.lat.notnull() & CAC_df.lng.notnull()]



# Convert lat/lon to (n, 2) nd-array format for heatmap

# Then send to list

CAC_loc = list(CAC_df[['lat', 'lng']].values)



# Add the location name to the markers

CAC_info = list(CAC_df["location_name"].values)



#Set up folium map

fol_map = folium.Map([41.8781, -87.6298], zoom_start=4)



# plot heatmap

heat_map = plugins.HeatMap(CAC_loc, name="COVID Testing Sites")

fol_map.add_child(heat_map)



# plot markers

markers = plugins.MarkerCluster(locations = CAC_loc, popups = CAC_info, name="Testing Site")

fol_map.add_child(markers)



#Add Layer Control

folium.LayerControl().add_to(fol_map)



fol_map