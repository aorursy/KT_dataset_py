# Configure matplotlib.

%matplotlib inline



# Unclutter the display.

import pandas as pd; 

pd.set_option('max_columns', 6)
import geopandas as gpd

import geoplot as gplt

usa_cities = gpd.read_file(gplt.datasets.get_path('usa_cities'))

usa_cities.head()
continental_usa_cities = usa_cities.query('STATE not in ["HI", "AK", "PR"]')

gplt.pointplot(continental_usa_cities)
usa_cities['STATE'].unique()
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))

gplt.polyplot(contiguous_usa)
ax = gplt.polyplot(contiguous_usa)

gplt.pointplot(continental_usa_cities, ax=ax)
import geoplot.crs as gcrs



ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())

gplt.pointplot(continental_usa_cities, ax=ax)
ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())

gplt.pointplot(continental_usa_cities, ax=ax)
ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())

gplt.pointplot(continental_usa_cities, ax=ax, hue='ELEV_IN_FT', legend=True)
continental_usa_cities
ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())

gplt.pointplot(continental_usa_cities, ax=ax, hue='ELEV_IN_FT', cmap='nipy_spectral', legend=True)
ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())

gplt.pointplot(continental_usa_cities, ax=ax, hue='ELEV_IN_FT', cmap='cool', legend=True)
ax = gplt.polyplot(

    contiguous_usa, projection=gcrs.AlbersEqualArea(),

    edgecolor='white', facecolor='lightgray',

    figsize=(12, 8)

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
gplt.choropleth(

    contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),

    edgecolor='white', linewidth=2,

    cmap='cool', legend=True,

    scheme='FisherJenks',

    legend_labels=[

        '<3 million', '3-6.7 million', '6.7-12.8 million',

        '12.8-25 million', '25-37 million'

    ]

)
contiguous_usa
boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))

collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))



ax = gplt.kdeplot(collisions, cmap='Reds', shade=True, clip=boroughs, projection=gcrs.AlbersEqualArea())

gplt.polyplot(boroughs, zorder=1, ax=ax)
import pandas as pd;

pd.set_option('max_columns', 6) 

# Unclutter display.

import geopandas as gpd

import geoplot as gplt



# load the example data

nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))

nyc_boroughs
nyc_boroughs.geometry
nyc_boroughs.crs

nyc_map_pluto_sample = gpd.read_file(gplt.datasets.get_path('nyc_map_pluto_sample'))

nyc_map_pluto_sample
nyc_map_pluto_sample.crs = {'init': 'epsg:2263'}

nyc_map_pluto_sample.crs
nyc_map_pluto_sample
nyc_map_pluto_sample = nyc_map_pluto_sample.to_crs(epsg=4326)

nyc_map_pluto_sample
nyc_map_pluto_sample
type(nyc_boroughs.geometry.iloc[0])
type(nyc_map_pluto_sample.geometry.iloc[0])
%time gplt.polyplot(nyc_boroughs.geometry.map(lambda shp: shp.convex_hull))
nyc_boroughs
type(nyc_map_pluto_sample.geometry)
(nyc_map_pluto_sample.geometry).is_valid
(nyc_map_pluto_sample.geometry).geom_type
(nyc_map_pluto_sample.geometry).total_bounds
(nyc_map_pluto_sample.geometry).bounds
(nyc_map_pluto_sample.geometry).area
gpd.options
import fiona; help(fiona.open)
nyc_map_pluto_sample.info()
nyc_map_pluto_sample.isna().any() # Which column contains missing data
nyc_map_pluto_sample.isna().sum() # how many missing values in any column
nyc_map_pluto_sample.notna().any() # Which column contains missing data
nyc_map_pluto_sample.notna().any().sum() # Which column contains missing data
nyc_collisions_sample = pd.read_csv(gplt.datasets.get_path('nyc_collisions_sample'))

nyc_collisions_sample
from shapely.geometry import Point

collisions_points = nyc_collisions_sample.apply( 

    lambda srs: Point(float(srs['LONGITUDE']),float(srs['LATITUDE'])), 

    axis='columns')
collisions_points
import geopandas as gpd

nyc_collisions_sample_geocoded = gpd.GeoDataFrame(nyc_collisions_sample,geometry=collisions_points)
nyc_collisions_sample_geocoded
obesity = pd.read_csv(gplt.datasets.get_path('obesity_by_state'), sep='\t')

obesity.head()
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))

contiguous_usa.head()
result = contiguous_usa.set_index('state').join(obesity.set_index('State'))

result.head()
import geoplot.crs as gcrs

gplt.cartogram(result, scale='Percent',projection= gcrs.AlbersEqualArea())
nyc_boroughs.to_file('boroughs.geojson', driver='GeoJSON')
{

  "type": "Feature",

  "geometry": {

    "type": "Point",

    "coordinates": [125.6, 10.1]

  },

  "properties": {

    "name": "Dinagat Islands"

  }

}
%matplotlib inline

import geopandas as gpd

import geoplot as gplt



usa_cities= gpd.read_file(gplt.datasets.get_path('usa_cities'))

contiguous_usa=gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
usa_cities
usa_cities=usa_cities.query('STATE not in ["AK","HI","PR"]')
import geoplot.crs as gcrs

ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator())

gplt.pointplot(usa_cities,ax=ax

              )
import geoplot.crs as gcrs

ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator())

gplt.pointplot(usa_cities,ax=ax,hue='ELEV_IN_FT'

              )
import mapclassify as mc

scheme = mc.Quantiles(usa_cities['ELEV_IN_FT'],k=5)

ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator())

gplt.pointplot(usa_cities,ax=ax,hue='ELEV_IN_FT',scheme=scheme)
ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator())

gplt.pointplot(usa_cities,ax=ax,hue='ELEV_IN_FT',cmap='terrain')
ax=gplt.webmap(contiguous_usa,projection=gcrs.WebMercator())

gplt.pointplot(usa_cities,ax=ax,hue='ELEV_IN_FT',cmap='cool')
large_continental_usa_cities = usa_cities.query('POP_2010 > 100000')



ax = gplt.webmap(contiguous_usa, projection=gcrs.WebMercator())

gplt.pointplot(

    large_continental_usa_cities, projection=gcrs.AlbersEqualArea(),

    scale='POP_2010', limits=(4, 50),

    ax=ax

)