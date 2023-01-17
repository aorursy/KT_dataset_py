!pip install rtree

!pip install osmnx
!apt-get -y install libspatialindex-dev
import geopandas



cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))



world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

countries = world[['geometry', 'name']]

countries = countries.rename(columns={'name':'country'})



cities_with_country = geopandas.sjoin(cities, countries, how="inner", op='intersects')



cities_with_country.head()