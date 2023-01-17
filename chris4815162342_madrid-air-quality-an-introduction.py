import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString
path = r'../input/air-quality-madrid/stations.csv'
stations = pd.read_csv(path)
geometry = [Point(xy) for xy in zip(stations['lon'], stations['lat'])]
crs = {'init': 'epsg:4326'}
geoDF_stations = gpd.GeoDataFrame(stations, crs=crs, geometry=geometry)
geoDF_stations_new = geoDF_stations.to_crs({'init': 'epsg:25830'}) 
#streetsystem = gpd.read_file('../input/location-of-the-streets-and-stations/call2016.shp')
#calleselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Calle"]
#avdselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Avda"]
#ctraselected = streetsystem.loc[streetsystem['VIA_TVIA'] == "Ctra"]
#calleandavd = calleselected.append(avdselected)
#streetselected = calleandavd.append(ctraselected)
#base = geoDF_stations_new.plot(figsize=(32,20), marker='o',color='red',markersize=100.0,label='Stations');
#mapMadrid = streetselected.plot(figsize=(32,20), ax=base,color='blue', edgecolor='blue',markersize=0.01,label='Streets');
#plt.ylim((4465000,4485000))
#plt.xlim((430000,455000))
#plt.legend(loc = 'lower right', framealpha=1)
#plt.xlabel("Longitude")
#plt.ylabel("Latitude")
#plt.title("Madrid city center street map with measurement stations")
#plt.show(mapMadrid)