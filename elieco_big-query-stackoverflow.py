# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "stackoverflow" dataset

dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Get a list of available tables 

tables = list(client.list_tables(dataset))

list_of_tables = [t.table_id for t in tables]



# Print your answer

print(list_of_tables)
# Construct a reference to the "posts_answers" table

answers_table_ref = dataset_ref.table("users")



# API request - fetch the table

answers_table = client.get_table(answers_table_ref)



# Preview the first five lines of the "posts_answers" table

client.list_rows(answers_table, max_results=5).to_dataframe()
# Your code here

questions_query = """

                  SELECT *

                  FROM `bigquery-public-data.stackoverflow.users`

                  WHERE display_name LIKE '%Elie Constantine%'

                  """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

questions_query_job = client.query(questions_query, job_config=safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

questions_results = questions_query_job.to_dataframe() # Your code goes here



# Preview results

print(questions_results.head())