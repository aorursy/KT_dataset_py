# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
# query that select row doesn't contain ppm
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# return a result dataframe without exceeding 1GB
countries = open_aq.query_to_pandas_safe(query)
countries.country.unique()
# Your Code Goes Here
query_2 = """SELECT pollutant
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE value = 0
          """
zero_pollutant = open_aq.query_to_pandas_safe(query_2)
zero_pollutant.pollutant.unique()