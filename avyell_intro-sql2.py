# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # Any results you write to the current directory are saved as output.
from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()
dataset_ref = client.dataset("openaq", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
table_ref = dataset_ref.table('global_air_quality')

table = client.get_table(table_ref)
# table.schema
client.list_rows(table, max_results=5).to_dataframe()
# our first query
'''
query = 'SELECT BLAH FROM BLAH WHERE BLAH = BLAH'
in actual code query is w three apostrophes, not 1
'''


# Query to select all the items from the "city" column where the "country" column is 'US'
query = """
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
# so now we have a query
# we have to submit the query to the client

query_job = client.query(query)
us_cities = query_job.to_dataframe()
# What five cities have the most measurements?
us_cities.city.value_counts().head()
# we should scan the data (not fully)
# before processing it because it will be much more efficient

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
# # Only run the query if it's less than 1 GB
# ONE_GB = 1000*1000*1000
# safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)

# # Set up the query (will only run if it's less than 1 GB)
# safe_query_job = client.query(query, job_config=safe_config)

# # API request - try to run the query, and return a pandas DataFrame
# job_post_scores = safe_query_job.to_dataframe()

# # Print average score for job posts
# job_post_scores.score.mean()