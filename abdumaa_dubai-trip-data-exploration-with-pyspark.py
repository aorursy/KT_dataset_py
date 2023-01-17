#dont forgeet to run "pip install pyspark" in the console
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
sc = SparkContext(appName = "dubaitripdata")
spark = SparkSession.Builder().getOrCreate()
processed_file = "../input/tripgenerateddubai1/dubai.dot.month1_11000 9000 0.4 720 5000.csv"
!head "$processed_file"
!wc -l "$processed_file"
from pyspark.sql.types import *
schema = StructType([
                     StructField("tripId", IntegerType(), True),
                     StructField("taxiId", IntegerType(), True),
                     StructField("PU_timeStamp", TimestampType(), True),
                     StructField("PU_lat", DoubleType(), True),
                     StructField("PU_long", DoubleType(), True),
                     StructField("DO_timeStamp", TimestampType(), True),                 
                     StructField("DO_lat", DoubleType(), True),
                     StructField("DO_long", DoubleType(), True),
                     ])
df = spark.read.csv(processed_file, header = "True" ,schema= schema, mode="DROPMALFORMATED", sep=",", timestampFormat = "yyyyMMdd HH:mm:ss" )
df.printSchema()
print("the number of trips is ",df.count())
df.sample(10/df.count()).show()
import matplotlib.pyplot as plt
plt.title("raw PU geo points")
plt.scatter(df.select('PU_long').collect(), df.select('PU_lat').collect(), marker='.')
#how to plot geo data on OSM
# https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db
BBox = (df.select('PU_long').rdd.min()['PU_long'], df.select('PU_long').rdd.max()['PU_long'],
       df.select('PU_lat').rdd.min()['PU_lat'], df.select('PU_lat').rdd.max()['PU_lat'])
print(BBox)
#use the bbox coordinates to get the map png from OSM
ruh_m = plt.imread('../input/dubaiosmmap/dubai_osm.png')
fig, ax = plt.subplots(figsize = (10,10))
#df.select('PU_long').collect(), df.select('PU_lat').collect()
ax.scatter(df.select('PU_long').collect(), df.select('PU_lat').collect(), zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Plotting PU Geo Data on Dubai Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= 'equal')
df_duration = df.select(df.DO_timeStamp,df.PU_timeStamp).withColumn('trip_duration(min)', (df.DO_timeStamp.cast("long") - df.PU_timeStamp.cast("long"))/60)
df_duration.describe().show()
df_duration.select('trip_duration(min)').toPandas().hist(bins=100)
plt.title("trip duration (min) distrib\ncomments:avg=84min, skewed ")
df_taxi = df.groupBy(df.taxiId).count()
df_taxi.describe().show()
df_taxi.select('count').toPandas().hist(bins=range(0, 105))
plt.title("taxi's number of trip distrib\ncomment: ~400 taxis did only 1 trip per mont")

