# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
#Listimg all the tables in the "openaq" dataset

tables=list(client.list_tables(dataset))



# Print names of all tables in the dataset 

for table in tables:

    print(table.table_id)
table.schema
query = """

        SELECT city, country

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """
# Set up the query

query_job = client.query(query)
# Convert to dataframes

us_cities = query_job.to_dataframe()
us_cities

us_cities.city.value_counts().head()
query = """

        SELECT city, source_name

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """

query_job=client.query(query)

us_cities=query_job.to_dataframe()

us_cities.head()
query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = 'US'

        """

query_job=client.query(query)

us_cities=query_job.to_dataframe()

us_cities.head()
# Query to get the score column from every row where the type column has value "job"

query = """

        SELECT *

        FROM `bigquery-public-data.openaq.global_air_quality`

        WHERE country = "US" 

        """



# Create a QueryJobConfig object to estimate size of query without running it

dry_run_config = bigquery.QueryJobConfig(dry_run=True)



# API request - dry run query to estimate costs

dry_run_query_job = client.query(query, job_config=dry_run_config)



print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
# Only run the query if it's less than 100 MB

ONE_HUNDRED_MB = 100*1000*1000

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)



# Set up the query (will only run if it's less than 100 MB)

safe_query_job = client.query(query, job_config=safe_config)



# API request - try to run the query, and return a pandas DataFrame

safe_query_job.to_dataframe()