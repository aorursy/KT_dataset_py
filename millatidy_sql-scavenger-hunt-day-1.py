import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")
open_aq.list_tables()

open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")
query=""" 
        SELECT DISTINCT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE LOWER(unit) != "ppm" 
        ORDER BY country"""

open_aq.estimate_query_size(query)
open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
query2 = """
        SELECT DISTINCT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
zero_p = open_aq.query_to_pandas_safe(query2)
zero_p
