# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def displayWide(df):
    pd.set_option('display.max_colwidth', 0)
    display(df)
    pd.set_option('display.max_colwidth', 50)
def displayAll(df):
    pd.set_option('display.max_rows', None)
    display(df)
    pd.set_option('display.max_rows', 10)
def displayAllWide(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 0)
    display(df)
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_colwidth', 50)
from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset("google_analytics_sample", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))
for table in tables:  
    print(table.table_id)
# Construct a reference to the first table
table_ref = dataset_ref.table("ga_sessions_20160801")

# API request - fetch the table
table = client.get_table(table_ref)
#Run cell
#See shema output
schema = table.schema
schema
# Use autocomplete feature to see available fields:
# (Put cursor at the end of the following line then press tab)
#schema[0].
df_head = client.list_rows(table, max_results=5).to_dataframe()
# displayWide(df_head)  #Uncomment to see all fields 
df_head
# Query to estimate
query = """
SELECT fullVisitorId, totals.timeOnSite  
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 
"""

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
#Execute query and return dataframe

query_job = client.query(query)
df_result = query_job.to_dataframe()
df_result