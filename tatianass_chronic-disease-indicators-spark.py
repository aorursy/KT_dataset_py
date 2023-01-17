### Spark ###

# To find out where the pyspark

import findspark

findspark.init()



# Creating Spark Context

from pyspark import SparkContext

from pyspark.sql import functions as F

from pyspark.sql import SQLContext

from pyspark.sql import Window

from pyspark.sql.types import StructType, StringType, FloatType



### Working with data ###

import numpy as np

import pandas as pd



### Visualization ###

import altair as alt

import plotly.plotly as py

import plotly.graph_objs as go



### Utils ###

import datetime

from urllib.request import urlopen



# File format #

import csv

import json

import xml.etree.ElementTree as ET





#### Setup plotly

py.sign_in(username='**your username**', api_key='**your key**')



#### Setup Spark

sc = SparkContext("local", "Testing")



spark = SQLContext(sc)



#### Load information

df = spark.read.csv("data/disease_indicators.csv", sep=',', header=True)



# Repartition the data by Topic

df2 = df.repartition('Topic')

df2.cache()

df2.rdd.getNumPartitions()



### Prepare Data

df_location = (df2

               .filter("GeoLocation IS NOT NULL AND Topic IS NOT NULL AND DataValue IS NOT NULL")

               .withColumn('DataValue', F.col('DataValue').cast('Float'))

              ).filter("DataValue IS NOT NULL")



#### Get top 10 cities by disease



### With Hive

df_location.createOrReplaceTempView('df')



top_10_cities = spark.sql("""

SELECT *

FROM (

    SELECT Topic, LocationDesc AS city, DataValue,

    DENSE_RANK() over (PARTITION BY Topic ORDER BY DataValue DESC) as dense_rank

    FROM df

)

WHERE dense_rank <=10

""")



### With PySpark

# top_10_window = Window.partitionBy('Topic').orderBy(F.col('DataValue').desc())

# top_10_cities = (

#     df_location.select('Topic', F.col('LocationDesc').alias('city'), 'DataValue', F.dense_rank().over(top_10_window).alias('rank')) # Using dense rank to get cities with similar positions

#     .filter(F.col('rank') <= 10)

# )



to_n_ranks_cities = (

    top_10_cities

    .groupBy('city')

    .agg(F.countDistinct('Topic').alias('Number of times in top 10'))

    .orderBy('Number of times in top 10') # Since the orientation is horizontal, the sort must be the inverse order of what I want

).toPandas()



### Testing Hypothesis

data = [go.Bar(

            y=to_n_ranks_cities['city'],

            x=to_n_ranks_cities['Number of times in top 10'],

            orientation = 'h',

            text=to_n_ranks_cities['Number of times in top 10']

    )]



layout = go.Layout(

    title='Frequency of Cities in top 10 ranking for diseases',

    titlefont=dict(size=20),

    width=1000,

    height=1400,

    yaxis=go.layout.YAxis(

        ticktext=to_n_ranks_cities['city'],

        tickmode='array',

        automargin=True

    )

)



fig = go.Figure(data=data, layout=layout)



py.iplot(fig, filename='cities-rank-frequency')