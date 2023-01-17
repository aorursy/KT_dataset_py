# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
openaq.list_tables()
openaq.table_schema("global_air_quality")
openaq.head("global_air_quality")
# this query looks in the global_air_quality table in the openaq
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be in GigaBytes
openaq.estimate_query_size(query)
# only run this query if it's less than 100 MB
no_ppm_countries = openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
no_ppm_countries['country'].unique()
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """

# only run this query if it's less than 100 MB
poll = openaq.query_to_pandas_safe(query2, max_gb_scanned=0.1)
poll['pollutant'].unique()
