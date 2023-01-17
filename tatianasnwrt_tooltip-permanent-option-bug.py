import geopandas as gdp

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



folium.__version__
# Load geo dataframe deom shape file (Budapest districts)

gdf_districts = gdp.read_file("../input/geo-bud-districts/Gdf_districts/Gdf_districts/gdf_districts.shp")

gdf_districts.crs = {'init' :'epsg:4326'}



# Add a column with district code for each row

for i in range(len(gdf_districts)):

    gdf_districts.loc[i,'district_code'] = i + 1



gdf_districts.head()
# Initiate a map

m_1 = folium.Map(location=[47.4979,19.0402], tiles='cartodbpositron', zoom_start=11)



# Pass necessary options to mykwargs variable

# mykwargs = {"fields": ['district_code']} # works

mykwargs = {"permanent": True, "fields": ['district_code']} # does not work

mytooltip = folium.GeoJsonTooltip(**mykwargs)



folium.GeoJson(

    gdf_districts[['geometry','district_code']], 

    tooltip=mytooltip

    ).add_to(m_1)



# Display the map

m_1