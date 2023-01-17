import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

injectionWellData = sqlContext.read.load('data/InjectionWells.csv',

                                  format='com.databricks.spark.csv',

                                  header='true',

                                  inferSchema='true')

injectionWellData = injectionWellData.drop('_c18','_c19','_c20')

earthquakeData = sqlContext.read.load('data/okQuakes.csv',

                                  format='com.databricks.spark.csv',

                                  header='true',

                                  inferSchema='true')
injectionWellData.describe().toPandas().transpose()
earthquakeData.describe().toPandas().transpose()
for dtype in injectionWellData.dtypes:

    print ('No of distinct values in column',dtype[0],'-',injectionWellData.select(dtype[0]).distinct().count(),'|','Datatype -',dtype[1])
injectionWellData.select('WellType').distinct().collect()
from pyspark.sql import functions as f

# output_format = 'MM/dd/yyyy'  # Some SimpleDateFormat string

# injectionWellData.select(date_format(

#     unix_timestamp("Approval Date", "MM/dd/yyyy").cast("timestamp"), 

#     output_format

# )).collect()



# injectionWellData.select(to_date(injectionWellData['Approval Date'], 'MM/dd/yyyy').alias('date')).dtypes



# injectionWellData = injectionWellData.withColumn("Approval_Date", to_date((injectionWellData['Approval Date']), "MM/dd/yyyy"))

# injectionWellData = injectionWellData.drop('Approval Date')
injectionWellData = injectionWellData.na.drop()
injectionWellData.dtypes
year_from, year_to = 1998,2000



wellSpan = injectionWellData.where((year(injectionWellData.Approval_Date)<year_to) & (year(injectionWellData.Approval_Date)>year_from)).select('Approval_Date','LAT','LONG')
from pyspark.sql.functions import *



injectionWellData.select([count(when(isnull(c), c)).alias(c) for c in injectionWellData.columns]).show()
for dtype in earthquakeData.dtypes:

    print ('No of distinct values in column',dtype[0],'-',earthquakeData.select(dtype[0]).distinct().count(),'|','Datatype -',dtype[1])
earthquakeData.select('time','place').distinct().take(10)
earthquake_from, earthquake_to = 1999,2002



earthquakeSpan = earthquakeData.where((year(earthquakeData.time)<earthquake_to) & (year(earthquakeData.time)>earthquake_from)).select('time','latitude','longitude','mag','place')
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

img = plt.imread('usamap.png', 0)

e = earthquakeSpan.toPandas()



padding = 5

min_lat,max_lat = earthquakeSpan.agg({"latitude": "min"}).collect()[0],earthquakeSpan.agg({"latitude": "max"}).collect()[0]

min_lat,max_lat = min_lat['min(latitude)']-padding,max_lat['max(latitude)']+padding

min_long,max_long = earthquakeSpan.agg({"longitude": "min"}).collect()[0],earthquakeSpan.agg({"longitude": "max"}).collect()[0]

min_long,max_long = min_long['min(longitude)']-padding,max_long['max(longitude)']+padding



min_mag,max_mag = earthquakeSpan.agg({"mag": "min"}).collect()[0],earthquakeSpan.agg({"mag": "max"}).collect()[0]

min_mag,max_mag = min_mag['min(mag)'],max_mag['max(mag)']



label = "Earthquake from "+str(earthquake_from)+" to "+str(earthquake_to)

# e.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)

ax = e.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),

                        label=label,

                        cmap=plt.get_cmap("jet"),

                       colorbar=False, alpha=0.4,

                      )

plt.imshow(img, extent=[min_long,max_long,min_lat,max_lat],alpha=0.1)

tick_values = np.linspace(min_mag, max_mag, 11)



plt.show()



earthquakeSpan.take(10)


i = wellSpan.toPandas()



i.plot(kind="scatter", x="LONG", y="LAT", alpha=0.4)

plt.show()
earthquakeSpan.select(min(earthquakeSpan.mag)).collect()[0]
import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon

from matplotlib.lines import Line2D



%matplotlib inline
usamap = gpd.read_file('states_21basic/states.shp')



e = earthquakeSpan.toPandas()

i = wellSpan.toPandas()



eLabel = "Earthquake from "+str(earthquake_from)+" to "+str(earthquake_to)

iLabel = 'Wells injected from '+str(year_from)+' to '+str(year_to)



# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



# # We restrict to South America.

# ax = world[world.continent == 'North America'].plot(

#     color='white', edgecolor='black' , figsize = (15,15))



crs = {'init' : 'epsg4326'}

geometry = [Point(xy) for xy in zip(e['longitude'],e['latitude'])]

well_geometry = [Point(xy) for xy in zip(i['LONG'],i['LAT'])]

unfilled_markers = [m for m, func in Line2D.markers.items()

                    if func != 'nothing' and m not in Line2D.filled_markers]





geo_df = gpd.GeoDataFrame(e,crs = crs, geometry = geometry)

geo = gpd.GeoDataFrame(i,crs = crs, geometry = well_geometry)

fig,ax = plt.subplots(figsize = (35,15))

usamap.plot(ax = ax , alpha = 0.9, color = 'grey' )

geo_df[geo_df['mag']>1].plot(ax = ax , markersize = 20, color = 'red',label = eLabel)

geo.plot(ax = ax , markersize = 10, color = 'blue' ,marker = '+',label = iLabel)

plt.legend(prop={'size':15})
geo_df[geo_df['mag']>1]
unfilled_markers = [m for m, func in Line2D.markers.items()

                    if func != 'nothing' and m not in Line2D.filled_markers]

unfilled_markers.index('+')
df = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

ax = df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
from mpl_toolkits.basemap import Basemap