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
#To use BigQuery, we'll import the Python package below:

from google.cloud import bigquery
#The first step in the workflow is to create a Client object. As you'll soon see, this Client object will play a central role in retrieving information from BigQuery datasets.

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset



dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset



tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
# Construct a reference to the "full" table

table_ref = dataset_ref.table("full")



# API request - fetch the table



table = client.get_table(table_ref)
table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
from google.cloud import bigquery



client = bigquery.Client()

dataset_ref2 = client.dataset("chicago_crime", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref2)
tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)
table_ref = dataset_ref2.table("crime")

table = client.get_table(table_ref)
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
# Preview the first five entries in the "by" column of the "full" table

client.list_rows(table, selected_fields=table.schema[:4], max_results=5).to_dataframe()
from google.cloud import bigquery



client = bigquery.Client()



dataset_ref = client.dataset("openaq", project= "bigquery-public-data")



dataset = client.get_dataset(dataset_ref)



tables = list(client.list_tables(dataset))



for i in tables:

     print(i.table_id)



table_ref = dataset_ref.table("global_air_quality")



table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()
table.schema
query = """

SELECT DISTINCT city FROM `bigquery-public-data.openaq.global_air_quality`

WHERE country = 'US'

limit 10

"""
client.query(query).to_dataframe()
query = """

select city, pollutant from `bigquery-public-data.openaq.global_air_quality`

where country = 'IN'



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

df = safe_query_job.to_dataframe()
df.head()
df['city'].value_counts()
from google.cloud import bigquery
client = bigquery.Client()



dataset_ref = client.dataset("openaq", "")