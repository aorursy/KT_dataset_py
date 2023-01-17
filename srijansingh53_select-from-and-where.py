# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")



# print all the tables in this dataset (there's only one!)

open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset

open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the

# "country" column is "us"

query = """SELECT city

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """
# the query_to_pandas_safe will only return a result if it's less

# than one gigabyte (by default)

us_cities = open_aq.query_to_pandas_safe(query)
us_cities
# What five cities have the most measurements taken there?

us_cities.city.value_counts().head()
query2 = """SELECT city, country

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """

us_cities = open_aq.query_to_pandas_safe(query2)

us_cities
query3 = """SELECT *

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """

# ` ` used instead of '  ' in FROM address_of_dataset

us_cities = open_aq.query_to_pandas_safe(query3)

us_cities


from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



dataset_ref = client.dataset("openaq", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))

for table in tables:  

    print(table.table_id)
table_ref = dataset_ref.table("global_air_quality")

table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
table.schema

query = """SELECT city

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'  

        """





# if want to query where country starts with 'I'.....

query_like = """SELECT city

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country LIKE 'I%'  

        """

# Using regular expression- return all country that starts with S or s

query_regexp = """SELECT country,

                    REGEXP_CONTAINS(country, r"^[Ss].*") AS starts_with_s

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """
# Set up the query

query_job = client.query(query_regexp)
# Convert to dataframe

us_cities = query_job.to_dataframe()
us_cities
us_cities.city.value_counts().head()
query2 = """SELECT *

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = 'US'

        """

# Running a dry-test to know the memory the query will consume

dry_run_config = bigquery.QueryJobConfig(dry_run = True)

# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config = dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))

# Only run the query if it's less than 1000 bytes (only if it is not cached, ie, the query has been ran already)

max_size_bytes = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = max_size_bytes)



# set up the query (only if it is less than 1000 Bytes)

safe_query_job = client.query(query2, job_config=safe_config)

safe_query_job.to_dataframe()
