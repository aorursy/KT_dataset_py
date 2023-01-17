# Your code goes here :)
# import package with helper functions
import bq_helper as bq

# create a helper object for this dataset
open_aq=bq.BigQueryHelper(active_project="bigquery-public-data",
                                 dataset_name="openaq")
#print all the tables in this dataset
open_aq.list_tables()

open_aq.head("global_air_quality")

query1= """
           SELECT city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country= 'US'
        """
us_cities=open_aq.query_to_pandas_safe(query1)

us_cities.city.value_counts().head()

query2="""
          SELECT DISTINCT country
          FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE unit!= "ppm"
          ORDER BY country DESC
       """
notppms=open_aq.query_to_pandas_safe(query2)
print(notppms)

query3="""
          SELECT DISTINCT pollutant
          FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE value=0
          ORDER BY pollutant
       """
pollutantzero=open_aq.query_to_pandas_safe(query3)
print(pollutantzero)