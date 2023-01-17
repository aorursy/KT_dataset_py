

import numpy as np # linear algebra

import pandas as pd # data processing

data_paths = {}

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_paths[filename] = os.path.join(dirname, filename)

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pyspark
from pyspark import SparkContext, SparkConf

sc = SparkContext('local')

print(sc.version)
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Print my_spark

print(spark)
# Print the tables in the catalog

print(spark.catalog.listTables())
# Create pd_temp

pd_temp = pd.DataFrame(np.random.random(10))



# Create spark_temp from pd_temp

spark_temp = spark.createDataFrame(pd_temp)



# Examine the tables in the catalog

print(spark.catalog.listTables())



# Add spark_temp to the catalog

spark_temp.createOrReplaceTempView('temp')



# Examine the tables in the catalog again

print(spark.catalog.listTables())
# Read in the airports data

airports = spark.read.csv(data_paths['airports.csv'], header = True)



# Show the data

airports.show(10)
# Read in the airports data

flights = spark.read.csv(data_paths['flights.csv'], header = True)



# Show the data shape

print((flights.count(), len(flights.columns)))



# print the tables in catalog

print(spark.catalog.listTables())



# adding data into spark view for sql querying

flights.createOrReplaceTempView('flights')



# print the tables in catalog

print(spark.catalog.listTables())
# see all columns in the table 

print(flights.columns)
# Don't change this query

query = "SELECT AIRLINE, FLIGHT_NUMBER, TAIL_NUMBER, ORIGIN_AIRPORT, DESTINATION_AIRPORT, SCHEDULED_DEPARTURE FROM flights LIMIT 5"



# Get the first 10 rows of flights

flights5 = spark.sql(query)



# Show the results

flights5.show()
# Don't change this query

query = "SELECT ORIGIN_AIRPORT, DESTINATION_AIRPORT, COUNT(*) as N FROM flights GROUP BY ORIGIN_AIRPORT, DESTINATION_AIRPORT"



# Run the query

flight_counts = spark.sql(query)



# Convert the results to a pandas DataFrame

pd_counts = flight_counts.toPandas()



# Print the head of pd_counts

print(pd_counts.head())
# Create the DataFrame flights

flights = spark.table("flights")



# Add duration_hrs

flights = flights.withColumn('duration_hrs', flights.AIR_TIME/60.)



# Show the head

flights.select('duration_hrs').show(10)
# Filter flights by passing a string

long_flights1 = flights.filter("DISTANCE > 1000")



# Filter flights by passing a column of boolean values

long_flights2 = flights.filter(flights.DISTANCE > 1000)
# Select the first set of columns

selected1 = flights.select('TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',)



# Select the second set of columns

temp = flights.select(flights.ORIGIN_AIRPORT, flights.DESTINATION_AIRPORT, flights.AIRLINE)



temp.show()
# Define first filter

filterA = flights.ORIGIN_AIRPORT == "SEA"



# Define second filter

filterB = flights.DESTINATION_AIRPORT == "PDX"



# Filter the data, first by filterA then by filterB

selected2 = temp.filter(filterA).filter(filterB)
# Define avg_speed

avg_speed = (flights.DISTANCE/(flights.AIR_TIME/60)).alias("avg_speed")



# Select the correct columns

speed1 = flights.select('TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', avg_speed)



# Create the same table using a SQL expression

speed2 = flights.selectExpr('TAIL_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', "DISTANCE/(AIR_TIME/60) as avg_speed")
# Cast the columns to integers

flights = flights.withColumn("MONTH", flights.MONTH.cast("integer"))

flights = flights.withColumn("DAY_OF_WEEK", flights.DAY_OF_WEEK.cast("integer"))

flights = flights.withColumn("AIR_TIME", flights.AIR_TIME.cast("integer"))

flights = flights.withColumn("DISTANCE", flights.DISTANCE.cast("double"))

flights = flights.withColumn("ARRIVAL_DELAY", flights.ARRIVAL_DELAY.cast("integer"))
# Find the shortest flight from PDX in terms of distance

flights.filter(flights.ORIGIN_AIRPORT == 'PDX').groupBy().min('DISTANCE').show()



# Find the longest flight from SEA in terms of air time

flights.filter(flights.ORIGIN_AIRPORT == 'SEA').groupBy().max('AIR_TIME').show()
# Group by tailnum

by_plane = flights.groupBy("TAIL_NUMBER")



# Number of flights each plane made

by_plane.count().show(10)



# Group by origin

by_origin = flights.groupBy("ORIGIN_AIRPORT")



# Average duration of flights from PDX and SEA

by_origin.avg("AIR_TIME").show(10)
# Import pyspark.sql.functions as F

import pyspark.sql.functions as F



# cast

flights = flights.withColumn("DEPARTURE_DELAY", flights.DEPARTURE_DELAY.cast("integer"))



# Group by month and dest

by_month_dest = flights.groupBy('MONTH', 'DESTINATION_AIRPORT')



# Average departure delay by month and destination

by_month_dest.avg('DEPARTURE_DELAY').show(10)



# Standard deviation of departure delay

by_month_dest.agg(F.stddev('DEPARTURE_DELAY')).show(10)
print(airports.columns)



# Examine the data

print(airports.show(10))
# Rename the faa column

airports = airports.withColumnRenamed("IATA_CODE", "DESTINATION_AIRPORT")



# Join the DataFrames

flights_with_airports = flights.join(airports , on = 'DESTINATION_AIRPORT', how = 'leftouter')



# Examine the new DataFrame

print(flights_with_airports.columns)

print(flights_with_airports.count())
flights_with_airports.select('FLIGHT_NUMBER', 'AIRPORT', 'CITY', 'STATE', 'COUNTRY', 'LATITUDE', 'LONGITUDE').show(10)
# Read in the airports data

airlines = spark.read.csv(data_paths['airlines.csv'], header = True)



# Show the data shape

print((airlines.count(), len(airlines.columns)))



airlines.show()
# filtering columns

model_data = flights.select('MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'TAIL_NUMBER', 'DESTINATION_AIRPORT', 'AIR_TIME', 'DISTANCE', 'ARRIVAL_DELAY',)



# Remove missing values

model_data = model_data.filter("ARRIVAL_DELAY is not NULL and AIRLINE is not NULL and AIR_TIME is not NULL and TAIL_NUMBER is not NULL")



# rows left

model_data.count()
# Create is_late (label)

model_data = model_data.withColumn("is_late", model_data.ARRIVAL_DELAY > 0)



# cast

model_data = model_data.withColumn("is_late", model_data.is_late.cast("integer"))



# rename column

model_data = model_data.withColumnRenamed("is_late", 'label')
model_data.show(15)
print('Labels distrubution:')

model_data.groupBy('label').count().show()
from pyspark.ml.feature import OneHotEncoder, StringIndexer

from pyspark.ml.feature import HashingTF, IDF, Tokenizer



# Create a StringIndexer

airline_indexer = StringIndexer(inputCol="AIRLINE", outputCol="airline_index")



# Create a OneHotEncoder

airline_encoder = OneHotEncoder(inputCol="airline_index", outputCol="airline_fact")
# Create a StringIndexer

dest_indexer = StringIndexer(inputCol="DESTINATION_AIRPORT", outputCol="dest_index")



# Create a OneHotEncoder

dest_encoder = OneHotEncoder(inputCol="dest_index", outputCol="dest_fact")
# Create a StringIndexer

tail_indexer = StringIndexer(inputCol="TAIL_NUMBER", outputCol="tail_index")



# Create a OneHotEncoder

tail_encoder = OneHotEncoder(inputCol="tail_index", outputCol="tail_fact")
from pyspark.ml.feature import VectorAssembler



# Make a VectorAssembler of 'MONTH', 'DAY_OF_WEEK', 'AIR_TIME', 'DISTANCE', 'ARRIVAL_DELAY','AIRLINE', 'TAIL_NUMBER', 'DESTINATION_AIRPORT'

vec_assembler = VectorAssembler(inputCols=["MONTH", "DAY_OF_WEEK", "AIR_TIME", "DISTANCE", "airline_fact", "dest_fact", "tail_fact"], outputCol="features")
# Import Pipeline

from pyspark.ml import Pipeline



# Make the pipeline

flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, airline_indexer, airline_encoder, tail_indexer, tail_encoder, vec_assembler])
piped_data = flights_pipe.fit(model_data).transform(model_data)
train_data, test_data = piped_data.randomSplit([.7, .3])
print('data points(rows) in train data :',  train_data.count())

print('data points(rows) in train data :',  test_data.count())
# Import LogisticRegression

from pyspark.ml.classification import LogisticRegression



# Create a LogisticRegression Estimator

lr = LogisticRegression()
# Import the evaluation submodule

import pyspark.ml.evaluation as evals



# Create a BinaryClassificationEvaluator

evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")
# Import the tuning submodule

import pyspark.ml.tuning as tune



# Create the parameter grid

grid = tune.ParamGridBuilder()



# Add the hyperparameter

grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))

grid = grid.addGrid(lr.elasticNetParam, [0, 1])



# Build the grid

grid = grid.build()
# Create the CrossValidator

cv = tune.CrossValidator(estimator=lr,

               estimatorParamMaps=grid,

               evaluator=evaluator)
# Call lr.fit()

best_lr = lr.fit(train_data)



# Print best_lr

print(best_lr)
# Use the model to predict the test set

test_results = best_lr.transform(test_data)



# Evaluate the predictions

print(evaluator.evaluate(test_results))