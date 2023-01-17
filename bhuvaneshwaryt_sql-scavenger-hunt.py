import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")
query = """SELECT country, location
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be
open_aq.estimate_query_size(query)
open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
job_post_scores = open_aq.query_to_pandas_safe(query)
job_post_scores.to_csv("pollution_unit_not_ppm.csv")
query2 = """SELECT country, location
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00 """

# check how big this query will be
open_aq.estimate_query_size(query2)
open_aq.query_to_pandas_safe(query2, max_gb_scanned=0.1)
job_post_scores2 = open_aq.query_to_pandas_safe(query2)
job_post_scores2.to_csv("pollution_value_zero.csv")