import geopandas as gpd
# Read in the data

full_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_lands/DEC_lands/DEC_lands.shp")

full_data.head()
type(full_data)
data = full_data.loc[:, ["CLASS", "COUNTY", "geometry"]].copy()
# How many lands of each type are there?

data.CLASS.value_counts()
# Select lands that fall under the "WILD FOREST" or "WILDERNESS" category

wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()

wild_lands.head()
wild_lands.plot()
wild_lands.geometry.head()
# Campsites in New York state (Point)

POI_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_pointsinterest/DEC_pointsinterest/Decptsofinterest.shp")

campsites = POI_data.loc[POI_data.ASSET=='PRIMITIVE CAMPSITE'].copy()



# Foot trails in New York state (LineString)

roads_trails = gpd.read_file("../input/geospatial-learn-course-data/DEC_roadstrails/DEC_roadstrails/Decroadstrails.shp")

trails = roads_trails.loc[roads_trails.ASSET=='FOOT TRAIL'].copy()



# County boundaries in New York state (Polygon)

counties = gpd.read_file("../input/geospatial-learn-course-data/NY_county_boundaries/NY_county_boundaries/NY_county_boundaries.shp")
ax = counties.plot(figsize=(10,10), color='none', edgecolor='gainsboro', zorder=3)

wild_lands.plot(color='lightgreen', ax=ax)

campsites.plot(color='maroon', markersize=2, ax=ax)

trails.plot(color='black', markersize=1, ax=ax)