# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset

open_aq.list_tables()
# print the first 5 rows of the "global_air_quality" dataset

open_aq.head("global_air_quality")
# query to select all the items from the "city" and "country" column

query1 = """SELECT city, country

            FROM `bigquery-public-data.openaq.global_air_quality`

        """
# only run this query if it's less than 100 MB

query1_result = open_aq.query_to_pandas_safe(query1, max_gb_scanned=0.1)
query1_result
# query to select all items in "city", "pollutant" and "value" column which has 

# pollutant  value more than 100 and  have US as the country

query2 = """SELECT city, pollutant, value

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE value > 100 AND country='US'

          """
query2_result = open_aq.query_to_pandas_safe(query2)
query2_result