!pip install pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
rawflights = spark.read.csv('../input/flights-and-airports-data/raw-flight-data.csv', inferSchema=True, header=True)
rawflights.show(10)
rawflights.count()
rawflights.explain()
rawflights.describe().show()
rawflights.distinct().count()
rawflights.printSchema()
rawflights.columns
flights = spark.read.csv('../input/flights-and-airports-data/flights.csv', inferSchema=True, header=True)
flights.show(10)
airports = spark.read.csv('../input/flights-and-airports-data/airports.csv', inferSchema=True, header=True)
airports.show(10)
data = flights.select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("label")))
data.show(10)


