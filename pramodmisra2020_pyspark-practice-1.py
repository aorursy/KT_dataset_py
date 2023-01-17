# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



!pip install pyspark
#PySpark Package Import



from pyspark.sql import SparkSession

import pyspark.sql.functions as F

import time



from pyspark.sql.functions import monotonically_increasing_id,broadcast
my_spark = SparkSession.builder.getOrCreate()

my_spark
print('Version',my_spark.version)
%%time

flights = my_spark.read.csv('/kaggle/input/flight-delays/flights.csv',header=True)

airports = my_spark.read.csv('/kaggle/input/flight-delays/airports.csv',header=True)

airlines = my_spark.read.csv('/kaggle/input/flight-delays/airlines.csv',header=True)

flights.printSchema()
type(flights)
flights.select('DISTANCE').dtypes
flights = flights.withColumn('DISTANCE',flights['DISTANCE'].cast('integer'))

flights
flights.show(5)
airports.show(5)
airlines.show(5)
flights = flights.withColumn('duration_hrs',flights.AIR_TIME/60)

flights.show(5)
dist_flights = flights.filter('DISTANCE>1000')

dist_flights.show(1)
dist_col = dist_flights.select('YEAR','MONTH','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','AIR_TIME',

                               'DISTANCE')

dist_col.show(5)
dist_col.filter(dist_col.DESTINATION_AIRPORT=='PBI').show(5)
dist_col.filter(dist_col.ORIGIN_AIRPORT=='JFK').filter(dist_col.DESTINATION_AIRPORT=='PBI').show(5)
dist_col.selectExpr('YEAR','MONTH','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT',

                    'DESTINATION_AIRPORT','AIR_TIME','DISTANCE','DISTANCE/(AIR_TIME /60)as Average_Speed').show(5)
dist_col.count()
dist_col.filter(dist_col.ORIGIN_AIRPORT=='SEA').groupby('ORIGIN_AIRPORT').count().show()
#Converting Column Type using cast

dist_col=dist_col.withColumn('AIR_TIME',dist_col['AIR_TIME'].cast('integer'))

dist_col
#Minimum value

dist_col.select('ORIGIN_AIRPORT','DISTANCE','AIR_TIME').groupby().min('DISTANCE').show()
# Maximum Value



dist_col.select('DISTANCE').groupby().max().show()



dist_col.select('DISTANCE').groupby().avg().show()
dist_col.groupby().sum('DISTANCE').collect()[0][0]
dist_col.filter(dist_col.ORIGIN_AIRPORT=='SEA').groupby('DESTINATION_AIRPORT').count().show(5)
month_df = dist_col.groupBy('MONTH','ORIGIN_AIRPORT')

month_df.avg('DISTANCE').show(5)
month_df.agg(F.mean('DISTANCE')).show(5)
flights.rdd.getNumPartitions()
airports.select(airports['AIRPORT']).distinct().show(5)
airports.filter('length(AIRPORT)<15').show()
airports.filter(~F.column('AIRPORT').contains('Airport')).show()
airports.withColumn('State Name',F.when(airports.STATE=='TX','Texas')).show(5)
airports.withColumn('Flag',F.when(airports.STATE=='TX','Texas').when(airports.STATE=='GA','Georgio')

                    .otherwise('N/A')).show(5)
airports.withColumn('ID',monotonically_increasing_id()).show()
start_time = time.time()



dest_cache = flights.select('DESTINATION_AIRPORT').cache()

print('First Call to cache',dest_cache,time.time()-start_time)

second_time = time.time()

print('Second Call to the dataframe',dest_cache,time.time()-second_time)
print('Is the dataframe Cached?',dest_cache.is_cached)

dest_cache.unpersist()



print('Is the dataframe Cached?',dest_cache.is_cached)
start_time = time.time()

df = airports.join(flights,airports['IATA_CODE']==flights['ORIGIN_AIRPORT'])

print('Time to Join the dataframe',time.time()-start_time)

df.explain()
start_time = time.time()

df_broadcast = airports.join(broadcast(flights),airports['IATA_CODE']==flights['ORIGIN_AIRPORT'])

print('Time to execute',start_time-time.time())
df_broadcast.explain()
airports.createOrReplaceTempView('Airports_tbl')



my_spark.sql('Select * from Airports_tbl').show()
my_spark.sql('select * from airports_tbl where state="PA"').show()