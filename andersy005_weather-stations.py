import warnings
warnings.filterwarnings('ignore')
from google.cloud import bigquery
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (16, 20)
import pandas_profiling
import cartopy.crs as ccrs
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# Use  bq_helper to create a BigQueryHelper object
noaa_gsod = BigQueryHelper(active_project= "bigquery-public-data", 
                              dataset_name= "noaa_gsod")

noaa_gsod.list_tables()
noaa_gsod.table_schema("stations")
%%time
noaa_gsod.head("stations", num_rows=20)
QUERY = """SELECT name, country, lat, lon, elev, begin, end
            FROM `bigquery-public-data.noaa_gsod.stations` """
noaa_gsod.estimate_query_size(QUERY)
QUERY = """SELECT name, country, lat, lon, elev, begin, `end`
            FROM `bigquery-public-data.noaa_gsod.stations` """
noaa_gsod.estimate_query_size(QUERY)
stations = noaa_gsod.query_to_pandas_safe(QUERY, max_gb_scanned=0.1)
stations.head()
stations.info()
stations.describe(include=["O"])
stations.describe()
pandas_profiling.ProfileReport(stations)
from shapely.geometry import Point
import geopandas as gpd
# Geopandas = pandas + (shapely * projection)
gdf = gpd.GeoDataFrame(stations, geometry=None)\
         .set_geometry([Point(r.lon, r.lat) for _, r in stations.iterrows()],
                       crs={"init": "EPSG:4326"})
gdf.head()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot();
# Now we can overlay stations over country outlines
base = world.plot(color='white')
plt.title('9000 Weather Stations')
gdf.plot(ax=base, marker='*', color='royalblue', markersize=5, alpha=0.5);