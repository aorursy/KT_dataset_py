import bq_helper
global_air_quality = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
global_air_quality.table_schema("global_air_quality")
query1 = """SELECT distinct(country) FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit!= "ppm" """
query2 = "SELECT distinct(pollutant) FROM `bigquery-public-data.openaq.global_air_quality` global_air_quality where value=0"
global_air_quality.estimate_query_size(query1)

global_air_quality.estimate_query_size(query2)
global_air_quality.query_to_pandas_safe(query1, max_gb_scanned=0.1)
global_air_quality.query_to_pandas_safe(query2, max_gb_scanned=0.1)