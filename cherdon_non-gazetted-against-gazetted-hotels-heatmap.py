import pandas as pd
import folium
from folium import plugins
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv("../input/hotels-gazetted.csv")
dataset.head()
gazetted = dataset.loc[dataset["Gazetted"]=="G"]
nongazetted = dataset.loc[dataset["Gazetted"]=="NG"]

latlong_G = gazetted[['Y', 'X']].values
latlong_NG = nongazetted[['Y','X']].values
def radians(deg):
    return deg*(np.pi/180)

def degrees(rad):
    return rad*(180/np.pi)

# Convert spherical coordinates into cartesian coordinates
def gps_to_vector(latlong):
    lat = radians(latlong[0])   # Convert to radians
    long = radians(latlong[1])  # Convert to radians
    x = np.cos(lat)*np.cos(long)
    y = np.cos(lat)*np.sin(long)
    z = np.sin(lat)
    return x, y, z

# Find average of latitude longitude points in the cartesian format (x,y,z)
def avg_vector(ls_latlong):
    x_total, y_total, z_total = [], [], []
    total_hotels = len(ls_latlong)
    for i in range(total_hotels):
        x, y, z = gps_to_vector(ls_latlong[i])
        x_total.append(x), y_total.append(y), z_total.append(z)
    avg_x, avg_y, avg_z = np.mean(x_total), np.mean(y_total), np.mean(z_total)
    return avg_x, avg_y, avg_z

# Find average of latitude longitude points in the cartesian format (x,y,z)
def avg_latlong(ls_latlong):
    x, y, z = avg_vector(ls_latlong)
    r = np.sqrt(x*x + y*y + z*z)
    long = degrees(math.atan2(y,x))     # Convert to degrees
    lat = degrees(math.atan2(z, r))     # Convert to degrees
    return lat, long

# Distance between two latlongs (arc length)
def haversine(latlong1, latlong2):
    dlat = radians(latlong2[0]) - radians(latlong1[0])
    dlong = radians(latlong2[1]) - radians(latlong1[1])
    a = np.sin(dlat/2)**2 + np.cos(radians(latlong2[0])) * np.cos(radians(latlong1[0])) * np.sin(dlong/2)**2
    c = 2 * math.asin(np.sqrt(a))
    radius = 6371000
    return c * radius

# Do we need to know the exact point of neighbour, or only distance for the algorithm?
def single_predict(target, training, k):
    distances = []
    sum = 0
    for i in range(len(training)):
        distance = haversine(training[i], target)
        distances.append(distance)
    distances = sorted(distances)
    for i in range(k):
        sum += distances[i]
    avg = (sum/k)*(-1)      # Average distance between point and k nearest neighbours, and inverse to MinMaxScale later
    return avg

def predict(test, training, lower_bound, upper_bound, k):
    distances = []
    for i in range(len(test)):
        distances.append([single_predict(test[i],training,k)])
    distances = np.array(distances)
    scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
    scaled_dist = scaler.fit_transform(distances)
    return scaled_dist
mapHeat = folium.Map(location=[1.3521, 103.8198], zoom_start = 11)
mapHeat.add_child(plugins.HeatMap(latlong_G, radius=25))
epicenter = list(avg_latlong(latlong_G))
print("The epicenter is located at:" + str(epicenter))
folium.Marker(location=epicenter, popup="epicenter", icon=folium.Icon(color='red')).add_to(mapHeat)
mapHeat
mapMarker = folium.Map(location=[1.3521, 103.8198], zoom_start = 11)
mapMarker.add_child(plugins.HeatMap(latlong_G, radius=25))
for position in latlong_NG:
    folium.Marker(position).add_to(mapMarker)
mapMarker
# Variables
g_ratio = 0.8
lower_bound = 0.5
k = 6

predicted = predict(latlong_NG, latlong_G, lower_bound, g_ratio, k)
print(predicted)