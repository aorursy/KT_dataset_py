# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
query = """SELECT DISTINCT country, unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
           ORDER BY country
        """
non_ppm = open_aq.query_to_pandas_safe(query)
non_ppm
non_ppm.country.values
query = """SELECT DISTINCT pollutant, value
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
           ORDER BY pollutant
        """
zero_value_pollutant = open_aq.query_to_pandas_safe(query)
zero_value_pollutant