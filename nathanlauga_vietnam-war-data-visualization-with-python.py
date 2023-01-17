# Loading libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline
bombing_operation = pd.read_csv("../input/THOR_Vietnam_Bombing_Operations.csv")
aircraft_glossary = pd.read_csv("../input/THOR_Vietnam_Aircraft_Glossary.csv", encoding = "ISO-8859-1")
weapons_glossary = pd.read_csv("../input/THOR_Vietnam_Weapons_Glossary.csv", encoding = "ISO-8859-1")
print("----- SHAPE OF DATASET -----")
print("BOMBING OPERATION : ",bombing_operation.shape)
print("AIRCRAFT GLOSSARY : ",aircraft_glossary.shape)
print("WEAPONS GLOSSARY : ",weapons_glossary.shape)
bombing_operation.head(2)
aircraft_glossary.head(2)
weapons_glossary.head(2)
countries_mission = bombing_operation["COUNTRYFLYINGMISSION"]
count = countries_mission.value_counts()
countries, y = count.keys().tolist(), count.values

plt.figure(figsize=(15,10))
ax= sns.barplot(x=countries, y=y,palette = sns.cubehelix_palette(len(countries)))
plt.xlabel('Countries')
plt.ylabel('Number of mission')
plt.title('Number of mission by country')
mission_lat_lon = bombing_operation[['COUNTRYFLYINGMISSION', 'TGTLATDD_DDD_WGS84', 'TGTLONDDD_DDD_WGS84']]
mission_lat_lon = mission_lat_lon.rename(columns={"COUNTRYFLYINGMISSION": "country", "TGTLATDD_DDD_WGS84": "latitude", "TGTLONDDD_DDD_WGS84" : "longitude"})
mission_lat_lon = mission_lat_lon[pd.notnull(mission_lat_lon['latitude'])]
mission_lat_lon = mission_lat_lon[pd.notnull(mission_lat_lon['country'])]
print(mission_lat_lon.head())
mission_lat_lon['latitude'] = mission_lat_lon['latitude'].round(2)
mission_lat_lon['longitude'] = mission_lat_lon['longitude'].round(2)
print("BEFORE DROP DUPLICATES : ",mission_lat_lon.shape)
mission_lat_lon = mission_lat_lon.drop_duplicates()
print("AFTER DROP DUPLICATES : ",mission_lat_lon.shape)
col = {}
for c in countries:
    if c == 'UNITED STATES OF AMERICA':
        col[c] = 'red'
    else:
        col[c] = 'blue'
    
print(col)
mission_lat_lon['colors'] = [col[c] for c in mission_lat_lon['country'].values]
print(mission_lat_lon.head())
fig = plt.figure(figsize=(15, 10))
m = Basemap(projection='lcc', resolution='h',
            width=5E6, height=5E6, 
            lat_0=16, lon_0=100)
m.etopo(scale=0.5, alpha=0.5)
m.drawcountries()
m.drawcoastlines()

lon = mission_lat_lon['longitude'].values
lat = mission_lat_lon['latitude'].values
col = mission_lat_lon['colors'].values

#Map (long, lat) to (x, y) for plotting
lons, lats = m(lon, lat)
# plot points as red dots
m.scatter(lons, lats, marker = 'o', color=col, s=1)
plt.show()