



import geopandas as gpd
full_data = gpd.read_file('../input/my-first-map-dec-lands/DEC_lands/DEClands.shp' )

full_data.head()
type(full_data)
data= full_data.loc[:, ["CLASS", "COUNTY","geometry"]].copy()

data.head()
data.CLASS.value_counts()
wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()

wild_lands.head()
wild_lands.plot()
# Campsites in New York state (Point)

POI_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_pointsinterest/DEC_pointsinterest/Decptsofinterest.shp")

campsites = POI_data.loc[POI_data.ASSET=='PRIMITIVE CAMPSITE'].copy()

campsites.head()
# Foot trails in New York state (LineString)

roads_trails = gpd.read_file("../input/geospatial-learn-course-data/DEC_roadstrails/DEC_roadstrails/Decroadstrails.shp")

trails = roads_trails.loc[roads_trails.ASSET=='FOOT TRAIL'].copy()



# County boundaries in New York state (Polygon)

counties = gpd.read_file("../input/geospatial-learn-course-data/NY_county_boundaries/NY_county_boundaries/NY_county_boundaries.shp")
#county precise the boundaries of a base map 
ax = counties.plot(figsize= (10,10), color= 'none', edgecolor='gainsboro', zorder=3)

wild_lands.plot(color='lightgreen', ax= ax)

campsites.plot(color= 'maroon', markersize=2, ax= ax)

trails.plot(color='red', markersize=1, ax=ax)