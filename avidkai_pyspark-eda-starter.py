!pip install pyspark
import os

import pandas as pd

import numpy as np



from pyspark import SparkConf, SparkContext

from pyspark.sql import SparkSession, SQLContext



from pyspark.sql.types import *

import pyspark.sql.functions as F
# Get a spark session or create one if not existing already

spark = SparkSession.builder.master("local").appName("COVID19-Prediction").getOrCreate()
# read csv into PySpark dataframe

df = spark.read.csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv', header=True)
# Shape of the Dataframe

print('( rows:', df.count(), ' columns:', len(df.columns),')')
# List the columns of the dataframe

df.columns
# Let's have a look the first 5 records

df.show(5)
# Generate descriptive stats of the data

df.describe().show()
# Check if there are any missing values

df.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in df.columns]).show()
# Show 5 countries with highest deaths

df.groupBy('Country').agg({'Deaths': 'Sum'}).sort('sum(Deaths)', ascending=False).show(5)
# Show 5 countries with highest deaths

df.groupBy('Country').agg({'Confirmed': 'Sum'}).sort('sum(Confirmed)', ascending=False).show(5)
# Show 5 countries with highest deaths

df.groupBy('Country').agg(F.sum('Recovered').alias('Total Recovered')).sort('Total Recovered', ascending=False).show(5)
# Show 5 countries with highest deaths

country_grouped_df = df.groupBy('Country').agg(F.sum('Confirmed').alias('Total Confirmed'), F.sum('Recovered').alias('Total Recovered'), F.sum('Deaths').alias('Total Deaths'))

country_grouped_df = country_grouped_df.withColumn('Recovery Percent', 100*F.col('Total Recovered') / F.col('Total Confirmed')).sort('Recovery Percent', ascending=False)

country_grouped_df.show(5)
country_grouped_df.select(['Country', 'Recovery Percent']).sort('Recovery Percent').toPandas().plot.barh(x='Country', figsize=(14, 10))
df = df.join(country_grouped_df, on = ['Country'])
df.show(2)
df = df.withColumn('Last Update', F.from_unixtime(F.unix_timestamp('Last Update', 'MM/dd/yyyy')).cast('date'))

df = df.withColumn('Date', F.from_unixtime(F.unix_timestamp('Date', 'MM/dd/yyyy')).cast('date'))
df = df.withColumn('month', F.month('Date'))

df = df.withColumn('year', F.year('Date'))

df = df.withColumn('dom', F.dayofmonth('Date'))

df = df.withColumn('dow', F.dayofweek('Date'))