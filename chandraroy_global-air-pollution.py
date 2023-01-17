# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For exam#
import bq_helper

# BigQuery Table bigquery-public-data.openaq.global_air_quality
global_air = bq_helper.BigQueryHelper(active_project ="bigquery-public-data",
                                     dataset_name ="openaq" )
# print a list of all the tables in the dataset i.e. openaq
global_air.list_tables()
# Only one table found in the dataset "openaq". 
# Let us print the schema of this table
global_air.table_schema("global_air_quality")
global_air.head("global_air_quality")
global_air.head("global_air_quality", selected_columns = "location", num_rows = 8)
#Check query size
query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            """


global_air.estimate_query_size(query)
# Actual running a query - query size upper cap 0.1 Gb
global_air.query_to_pandas_safe(query, max_gb_scanned = 0.1)
# Storing the dataframe returned from the query
country = global_air.query_to_pandas(query)
country
country.country.value_counts()
# Writing the output to a CSV file
country.to_csv("First_BigQuery.csv")