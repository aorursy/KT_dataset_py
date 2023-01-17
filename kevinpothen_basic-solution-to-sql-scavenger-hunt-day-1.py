# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# Code
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
c_unit_not_ppm = open_aq.query_to_pandas_safe(query)

c_unit_not_ppm.info()

print(c_unit_not_ppm)
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """
pollutants_zero = open_aq.query_to_pandas_safe(query)

pollutants_zero.info()

print(pollutants_zero)