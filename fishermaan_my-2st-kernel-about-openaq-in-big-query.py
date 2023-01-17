import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "openaq")
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
query = """SELECT DISTINCT country 
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE unit IS NOT NULL AND unit != 'ppm';"""

country = open_aq.query_to_pandas_safe(query= query)
country.country
query = """SELECT DISTINCT pollutant 
        FROM `bigquery-public-data.openaq.global_air_quality` 
        WHERE value IS NOT NULL AND value = 0;"""

pollutant = open_aq.query_to_pandas_safe(query= query)
pollutant