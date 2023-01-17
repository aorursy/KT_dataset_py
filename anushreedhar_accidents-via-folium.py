import pandas as pd
import numpy as np
import csv
df = pd.read_csv('Accidents_2005_to_2007.csv', dtype='unicode')
df.shape
df.columns
new_accidents_file = df[['Location_Easting_OSGR', 'Location_Northing_OSGR']].copy()

print(new_accidents_file.head(5))
import pandas as pd
df = pd.read_csv('Accidents_2005_to_2007.csv', dtype='unicode')
df = pd.read_csv('Accidents_2005_to_2007.csv', header = 0)

original_headers = list(df.columns.values)


# remove the non-numeric columns
df = df._get_numeric_data()
# put the numeric column names in a python list
numeric_headers = list(df.columns.values)
# create a numpy array with the numeric values for input into scikit-learn
numpy_array = df.as_matrix()
np.save("numpy_array",numpy_array)
np.load("numpy_array.npy")
from sklearn import datasets
from sklearn.svm import SVC
df.shape
clf = SVC()
from csv import reader
def load_csv(filename):
	data = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			data.append(row)
	return data
filename = 'Accidents_2005_to_2007.csv'
data = load_csv(filename)
print(filename)
print(data[0])
import pandas as pd
 
# reading csv file
data = pd.read_csv('Accidents_2005_to_2007.csv')
 
# shape of dataset
print("Shape:", data.shape)
 
# column names
print("\nFeatures:", data.columns)
 
# storing the feature matrix (X) and response vector (y)
X = data[data.columns[:-1]]
y = data[data.columns[-1]]
 
# printing first 5 rows of feature matrix
print("\nFeature matrix:\n", X.head())
 
# printing first 5 values of response vector
print("\nResponse vector:\n", y.head())
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
df = pd.read_csv('Accidents_2005_to_2007.csv', dtype='unicode')
import folium

district_map = folium.Map(location=[55, 0],
                   tiles='Mapbox Bright', zoom_start=5)
district_map
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from folium.plugins import MarkerCluster
m = folium.Map(location=[51.5, -0.1], zoom_start=10)
 
popup = "London"
london_marker = folium.Marker([51.5, -0.1], popup=popup)
 
m.add_child(london_marker)
 
m
accident_data = pd.read_csv("accidents_2005_to_2007.csv")
accident_data = accident_data.sample(n=500, random_state=42)
accident_data.dropna(subset=["Latitude", "Longitude"], inplace=True)

# First marker cluster
m = folium.Map(location=[51.5, -0.1], zoom_start=10)
 
locations = list(zip(accident_data.Latitude, accident_data.Longitude))
icons = [folium.Icon(icon="car", prefix="fa") for _ in range(len(locations))]
 
cluster = MarkerCluster(locations=locations, icons=icons)
m.add_child(cluster)
m
def get_geojson_grid(upper_right, lower_left, n=6):
    """Returns a grid of geojson rectangles, and computes the exposure in each section of the grid based on the vessel data.
     
    Parameters
    ----------
    upper_right: array_like
        The upper right hand corner of "grid of grids" (the default is the upper right hand [lat, lon] of the USA).
         
    lower_left: array_like
        The lower left hand corner of "grid of grids"  (the default is the lower left hand [lat, lon] of the USA).
         
    n: integer
        The number of rows/columns in the (n,n) grid.
         
    Returns
    -------
     
    list
        List of "geojson style" dictionary objects   
    """
     
    all_boxes = []
 
    lat_steps = np.linspace(lower_left[0], upper_right[0], n+1)
    lon_steps = np.linspace(lower_left[1], upper_right[1], n+1)
 
    lat_stride = lat_steps[1] - lat_steps[0]
    lon_stride = lon_steps[1] - lon_steps[0]
 
    for lat in lat_steps[:-1]:
        for lon in lon_steps[:-1]:
            # Define dimensions of box in grid
            upper_left = [lon, lat + lat_stride]
            upper_right = [lon + lon_stride, lat + lat_stride]
            lower_right = [lon + lon_stride, lat]
            lower_left = [lon, lat]
             
            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]
             
            geo_json = {"type": "FeatureCollection",
                        "properties":{
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features":[]}
 
            grid_feature = {
                "type":"Feature",
                "geometry":{
                    "type":"Polygon",
                    "coordinates": [coordinates],
                }
            }
 
            geo_json["features"].append(grid_feature)
             
            all_boxes.append(geo_json)
             
    return all_boxes
m = folium.Map(zoom_start = 5, location=[55, 0])
grid = get_geojson_grid(upper_right, lower_left, n=6)
 
for i, geo_json in enumerate(grid):
     
    color = plt.cm.Reds(i / len(grid))
    color = mpl.colors.to_hex(color)
     
    gj = folium.GeoJson(geo_json,
                        style_function=lambda feature, color=color: {
                                                                        'fillColor': color,
                                                                        'color':"black",
                                                                        'weight': 2,
                                                                        'dashArray': '5, 5',
                                                                        'fillOpacity': 0.55,
                                                                    })
    popup = folium.Popup("example popup {}".format(i))
    gj.add_child(popup)
     
    m.add_child(gj)
m
m = folium.Map(zoom_start = 5, location=[55, 0])
 
# Generate GeoJson grid
top_right = [58, 2]
top_left = [49, -8]
 
grid = get_geojson_grid(top_right, top_left, n=6)
 
# Calculate exposures in grid
popups = []
regional_counts = []
 
for box in grid:
    upper_right = box["properties"]["upper_right"]
    lower_left = box["properties"]["lower_left"]
 
    mask = (
        (accident_data.Latitude <= upper_right[1]) & (accident_data.Latitude >= lower_left[1]) &
        (accident_data.Longitude <= upper_right[0]) & (accident_data.Longitude >= lower_left[0])
           )
     
    region_incidents = len(accident_data[mask])
    regional_counts.append(region_incidents)
     
    total_vehicles = accident_data[mask].Number_of_Vehicles.sum()
    total_casualties = accident_data[mask].Number_of_Casualties.sum()
    content = "total vehicles {:,.0f}, total casualties {:,.0f}".format(total_vehicles, total_casualties)
    popup = folium.Popup(content)
    popups.append(popup)
 
worst_region = max(regional_counts)
     
# Add GeoJson to map
for i, box in enumerate(grid):
    geo_json = json.dumps(box)
     
    color = plt.cm.Reds(regional_counts[i] / worst_region)
    color = mpl.colors.to_hex(color)
 
    gj = folium.GeoJson(geo_json,
                        style_function=lambda feature, color=color: {
                                                                        'fillColor': color,
                                                                        'color':"black",
                                                                        'weight': 2,
                                                                        'dashArray': '5, 5',
                                                                        'fillOpacity': 0.55,
                                                                    })
     
    gj.add_child(popups[i])
    m.add_child(gj)
     
# Marker clusters
locations = list(zip(accident_data.Latitude, accident_data.Longitude))
icons = [folium.Icon(icon="car", prefix="fa") for _ in range(len(locations))]
 
# Create popups
popup_content = []
for incident in accident_data.itertuples():
    number_of_vehicles = "Number of vehicles: {} ".format(incident.Number_of_Vehicles)
    number_of_casualties = "Number of casualties: {}".format(incident.Number_of_Casualties)
    content = number_of_vehicles + number_of_casualties
    popup_content.append(content)
 
popups = [folium.Popup(content) for content in popup_content]
 
cluster = MarkerCluster(locations=locations, icons=icons, popups=popups)
m.add_child(cluster)
     
m.save("car_accidents.html")
