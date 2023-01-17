# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
#table_ref = open_aq.table('global_air_quality')
open_aq.head('global_air_quality', 10)
# Your Code Goes Here
query = """SELECT distinct country 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            """
open_aq.query_to_pandas_safe(query)
# Your Code Goes Here
# Your Code Goes Here
query = """SELECT distinct pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
            """
open_aq.query_to_pandas_safe(query)
