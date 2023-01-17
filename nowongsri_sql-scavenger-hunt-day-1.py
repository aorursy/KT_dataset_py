# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality", num_rows=5)
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
query = """SELECT country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
show_pollutants = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
query = """SELECT country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
show_pollutants = open_aq.query_to_pandas_safe(query)
show_pollutants.pollutant.value_counts().head()
# Your code goes here :)
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality", num_rows=5)
query = """SELECT distinct country, pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE pollutant != "pm25" and pollutant != "pm10"
            """
show_pollutants = open_aq.query_to_pandas_safe(query)
show_pollutants.country.count()



import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality", num_rows=5)
query = """SELECT distinct pollutant, value
           FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE value = 0
            """
show_value = open_aq.query_to_pandas_safe(query)
show_value
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality", num_rows=5)
query = """SELECT city, location, pollutant, value
           FROM `bigquery-public-data.openaq.global_air_quality`
          WHERE country = "TH" and pollutant ="pm10" 
          ORDER BY value
          """
            
show_TH = open_aq.query_to_pandas_safe(query)
show_TH