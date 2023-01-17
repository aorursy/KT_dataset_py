import bq_helper 
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query1 = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query1)
countries_no_ppm = open_aq.query_to_pandas_safe(query1)
countries_no_ppm.to_csv("countries_no_ppm.csv")
open_aq.table_schema("global_air_quality")
query2 = """SELECT distinct(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
open_aq.estimate_query_size(query2)
zero_pollutant = open_aq.query_to_pandas_safe(query2)
zero_pollutant.to_csv("zero_pollutant.csv")