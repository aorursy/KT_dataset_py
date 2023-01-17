# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the countries where the unit to measure pollution is something other 
# than ppm
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" 
            ORDER BY country
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollution_units = open_aq.query_to_pandas_safe(query)
# Look at the output of my query?
pollution_units
# query to select all the countries where the unit to measure pollution is something other 
# than ppm
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00 
            GROUP BY pollutant
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
zero_pollution = open_aq.query_to_pandas_safe(query2)
zero_pollution