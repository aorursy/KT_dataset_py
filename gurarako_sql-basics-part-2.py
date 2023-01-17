# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset

open_aq.list_tables()
# print the first 5 rows of the "global_air_quality" dataset

open_aq.head("global_air_quality")
# query to count number of different countries

query1 = """SELECT COUNT(DISTINCT(country))

            FROM `bigquery-public-data.openaq.global_air_quality`

        """
# only run this query if it's less than 100 MB

query1_result = open_aq.query_to_pandas_safe(query1, max_gb_scanned=0.1)
#check the query result which is now a dataframe

query1_result
# query to count number of different countries

query2 = """SELECT COUNT(DISTINCT(country)) AS number_of_countries

            FROM `bigquery-public-data.openaq.global_air_quality`

        """
# only run this query if it's less than 100 MB

query2_result = open_aq.query_to_pandas_safe(query2, max_gb_scanned=0.1)
#check the query result which is now a dataframe

query2_result
# query to select the country and its average value of pollutant

# columns that are not included within an aggregate function and must be included in the GROUP BY 

query3 = """SELECT country, AVG(value)

            FROM `bigquery-public-data.openaq.global_air_quality`

            GROUP BY country

            ORDER BY 2 DESC

          """
# only run this query if it's less than 100 MB

query3_result = open_aq.query_to_pandas_safe(query3, max_gb_scanned=0.1)
#check the query result which is now a dataframe

query3_result