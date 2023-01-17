import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium
art = pd.read_csv('../input/nashville-public-art/public_art.csv')
art.head()
neighborhoods = gpd.read_file('../input/nashville-nighborhoods/nashville_neighborhoods.geojson')
neighborhoods.head()
neighborhoods.loc[0, 'geometry']
print(neighborhoods.loc[0, 'geometry'])
# geopandas handles legend styling if you pass a dict of keywords
leg_kwds = {'title': 'Neighborhoods',
               'loc': 'upper left', 'bbox_to_anchor': (1, 1.03), 'ncol': 6}

neighborhoods.plot(column = 'name', legend = True, cmap = 'Set2', legend_kwds = leg_kwds)
plt.title('Neighborhoods')
plt.show();
art.columns = ['title', 'last_name', 'first_name', 'address', 'medium', 'type', 'desc', 'lat', 'lng', 'location']
art.head()
art['geometry'] = art.apply(lambda row: Point(row.lng ,row.lat), axis=1)
art.head()
type(art)
art_geo = gpd.GeoDataFrame(art, crs = neighborhoods.crs, geometry = art['geometry'])
type(art_geo)
neighborhood_art = gpd.sjoin(art_geo, neighborhoods, op = 'within')
neighborhood_art.head()
neighborhood_art[['name', 'title']].groupby('name').agg('count').sort_values(by = 'title', ascending = False)
urban_art = neighborhood_art.loc[neighborhood_art.name == 'Urban Residents']
urban_art.head()
urban_art.shape
urban_polygon = neighborhoods.loc[neighborhoods.name == 'Urban Residents']
urban_polygon.head() 
# define the plot as ax
ax = urban_polygon.plot(figsize = (12, 12), color = 'lightgreen')

# add the plot of urban_art to ax (the urban_polygon)
urban_art.plot(ax = ax, column = 'type', legend = True);
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show();
# look at the center of our urban_polygon
urban_polygon.geometry.centroid
# find the center of the urban polygon with the centroid property
center = urban_polygon.geometry.centroid

# get and store the first occurence which will be a Point geometry
center_point = center.iloc[0]

# print the types for center and center_point
print('center is :', type(center))
print('center_point is :', type(center_point))
# center point has longitude first
print(center_point)

# reverse the order when constructing the array for folium location
urban_center = [center_point.y, center_point.x]

# check the order of urban_center, the location we'll set for our folium map
print(urban_center)
# create our map of Nashville and show it
map_downtown = folium.Map(location =  urban_center, zoom_start = 15)
map_downtown
# show what iterrows() does
for row in urban_art.iterrows():
    row_values = row[1]
    print(row_values)
#draw our neighborhood: Urban Residents
folium.GeoJson(urban_polygon).add_to(map_downtown)

#iterate through our urban art to create locations and markers for each piece
#here lat is listed first!!
#also the apostrophe in the 4th row causes problems!

for row in urban_art.iterrows():
    row_values = row[1] 
    location = [row_values['lat'], row_values['lng']]
    popup = (str(row_values['title']) + ': ' + 
             str(row_values['type']) + '<br/>' +
             str(row_values['desc'])).replace("'", "`")
    marker = folium.Marker(location = location, popup = popup)
    
    marker.add_to(map_downtown)

#display our map
map_downtown
