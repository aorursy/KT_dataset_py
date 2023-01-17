"""
This is a notebook that illustrates how to use BigQuery to
retrieve data.
This is the content is from the SQL Scavenger hunt: Day 1 by Recheal Tatman
"""
#imports 
from bq_helper import BigQueryHelper
#create a BigQueryHelper object
open_aq = BigQueryHelper(active_project="bigquery-public-data",
                         dataset_name="openaq")

#print a list of tables in openAQ dataset                               )
open_aq.list_tables()
#print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
unit_query = """SELECT country 
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE unit != 'ppm'
             """
unit_query_data = open_aq.query_to_pandas_safe(unit_query)
#since the result is in pandas, you can use the pandas syntax to check for shape and other stuff.
unit_query_data.shape
zero_pollutant_query = """SELECT pollutant
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE value = 0
                        """
#query the data into pandas dataframe
zero_pollutant_data = open_aq.query_to_pandas_safe(zero_pollutant_query)
zero_pollutant_data.shape
