# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
not_ppm_query = """SELECT DISTINCT country, unit
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE LOWER(unit) != 'ppm';
"""

not_ppm_countries = open_aq.query_to_pandas_safe(not_ppm_query)
not_ppm_countries

value_zero_query = """SELECT DISTINCT pollutant
                        FROM `bigquery-public-data.openaq.global_air_quality`
                        WHERE value = 0;
"""
value_zero_pollutants = open_aq.query_to_pandas_safe(value_zero_query)
value_zero_pollutants