import geopandas as gpd



gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.head()
import matplotlib.pyplot as plt 

%matplotlib inline
gdf.plot()
gdf.plot(figsize=(15,10))

#We will put the size in inches and it is ordered by (Width and Height)
capital_cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

capital_cities.plot(figsize=(15,10), marker='d', color='blue', markersize=10)

capital_cities
countries= gdf.plot(figsize=(15,10),  edgecolor='black' )

capital_cities.plot(ax=countries, marker='D',color='red', markersize=10)
cities=capital_cities.plot(figsize=(15,10), marker='d', color='red', markersize=10)

#Reading the built in dataset for cities

ct=gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

#REading the built in data set for countries

wmap=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



#This creates a figure and attaches an axes on this

fig, ax = plt.subplots(figsize=(15,10))

ax.set_aspect('equal')

#These were the 2 steps that geopanda was doing in background



#World map layer

wmap.plot(ax=ax, edgecolor= 'black',color='white') #Check first argument it's different



#Cities layer

ct.plot(ax=ax, marker='o', color='red', markersize=5)# Check first argument length
# Importing Coordinate Reference System (CRS) from CartoPy

import cartopy.crs as ccrs



#Create a figure of certain size

plt.figure(figsize=(7.1299,6))



#Attaching the projection detail to the axes, we just have to use the projection name/title

ax=plt.axes(projection= ccrs.AlbersEqualArea())



 #Adding the coastlines, shape will be automatically created to the kind of projection we created

ax.coastlines()

#Shape, size and type of grid will be automatically choosed according to the projection we have selected

ax.gridlines()
plt.figure(figsize=(12,6))

ax=plt.axes(projection= ccrs.Geostationary())

ax.coastlines(color='blue')

ax.gridlines()
import geoplot as gplt

import geoplot.crs as gcrs
ny_population = gpd.read_file('https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/ny-census-partial.geojson')
ny_population.head()
usa= gpd.read_file('https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/contiguous-usa.geojson')

usa_cities =gpd.read_file('https://raw.githubusercontent.com/ResidentMario/geoplot-data/master/usa-cities.geojson')
usa.head()
usa_cities.head()
# we can get the percentage by dividing the white population by total



percent_white = ny_population['WHITE']/ ny_population['POP2000']

percent_white
#This function which plots a choropleth may takes many options



#First option is always the Geopandas dataframe(nyc_population in our case)

#'hue' parameter tells for which variable the plot is being plotted.

#'projection' parameter tells what kind of projection we want

# 'cmap' tells what color we want to show

#'linewidth' is simply the width of line dividing different shades.

#'edgecolor' tells the color of the lines of "linewidth"

#'k' stands for how many colors to be used

#"legend" is used to show whether it is legend or not i.e. TRUE or FALSE



gplt.choropleth(ny_population,figsize=(10,6), hue=percent_white,

                projection=gcrs.AlbersEqualArea(), cmap='Purples', linewidth=0.5, edgecolor='yellow', legend= True)





plt.title("Percentage white residents, 2000")
roads = gpd.read_file('https://github.com/ResidentMario/geoplot-data/raw/master/dc-roads.geojson')

roads.head(3)
#Sankey funtion arguments

#first argument is always going to be a dataframe

#"paths" parameters requires the special geometry columntalked about

#"scale" is the type of thickness of the roads/streets in string format

#"limits" tells the min and max of "scale"


ax = gplt.sankey(roads, projection=gcrs.Mollweide(),color='black',scale='aadt', limits=(0.1,10))

#ax=gplt.sankey(roads, figsize=(15,15), path=roads.geometry, projection=gcrs.Mollweide(), scale='aadt', limits=(0.1,10))

plt.title("Streets in washington DC by Average Daily Traffic, 2015")

#There is a dataset provided by the Geopandas called as "New York Boroughs Boundary Data"

#layer 2

nybbdata= gpd.read_file(gpd.datasets.get_path('nybb')).to_crs(epsg='4326')
nybbdata
#Injurious collisions/ accidents data of new york

#Layer 3

injurious_collisions=gpd.read_file('https://github.com/ResidentMario/geoplot-data/raw/master/nyc-injurious-collisions.geojson')



injurious_collisions
#Geoplot will automatically call inbuilt matplotlib to create 1 layer

#As usual first option to be used is always GeoDataFrame

#"shade" parameter tells whether to just draw lines or fill them with the shades

#"shade lowest" parameterscan turn on/off of the presence of low intensity shades



ax= gplt.kdeplot(injurious_collisions.sample(1000),figsize=(12,8), shade=True, color='yellow',shade_lowest=False,clip=nybbdata.geometry)

gplt.polyplot(nybbdata,ax=ax,zorder=1,edgecolor= 'black')


ax=gplt.kdeplot(injurious_collisions.sample(1000),figsize=(12,8), clip=nybbdata.geometry,shade=False, color='red',shade_lowest=True,cbar=True)

gplt.polyplot(nybbdata,ax=ax,edgecolor= 'black')
injurious_collisions
max(injurious_collisions['NUMBER OF PERSONS INJURED'])
injurious_collisions['NUMBER OF PERSONS INJURED']
gplt.polyplot(roads,ax=ax,edgecolor= 'black')
gplt.polyplot(nybbdata)
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
gplt.polyplot(contiguous_usa)
continental_usa_cities = usa_cities.query('STATE not in ["HI", "AK", "PR"]')

gplt.pointplot(continental_usa_cities)
ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator(),edgecolor='black')

gplt.pointplot(continental_usa_cities,ax=ax,marker=1)
ax=gplt.polyplot(contiguous_usa,projection=gcrs.WebMercator(),edgecolor='black')

gplt.pointplot(continental_usa_cities,ax=ax,marker=1)
continental_usa_cities
ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator(),edgecolor='black')

gplt.pointplot(continental_usa_cities,ax=ax,marker=1,hue='ELEV_IN_FT',legend=True)
ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator(),edgecolor='black')

gplt.pointplot(continental_usa_cities,ax=ax,marker=1,hue='ELEV_IN_FT',cmap='terrain',legend=True)
ax = gplt.polyplot(

    contiguous_usa, projection=gcrs.AlbersEqualArea(),

    edgecolor='white', facecolor='lightgray',

    figsize=(12, 18)

)

gplt.pointplot(

    continental_usa_cities, ax=ax, hue='ELEV_IN_FT', cmap='Blues',

    scheme='quantiles',

    scale='ELEV_IN_FT', limits=(1, 10),

    legend=True, legend_var='scale',

    legend_kwargs={'frameon': False},

    legend_values=[-110, 1750, 3600, 5500, 7400],

    legend_labels=['-110 feet', '1750 feet', '3600 feet', '5500 feet', '7400 feet']

)

ax.set_title('Cities in the Continental United States by Elevation', fontsize=16)
ax = gplt.polyplot(nybbdata, projection=gcrs.AlbersEqualArea())

gplt.kdeplot(injurious_collisions.sample(1000), n_levels=20, cmap='Reds', ax=ax,cbar=True,shade=False)