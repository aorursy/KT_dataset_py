# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#query that returns all countries where unit is not ppm
query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
         """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
not_ppm = open_aq.query_to_pandas_safe(query1)

#display results
not_ppm

#query to retrieve pollutants that have value of 0
query2 =  """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
         """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
zero_pollutant = open_aq.query_to_pandas_safe(query2)

#display results
zero_pollutant