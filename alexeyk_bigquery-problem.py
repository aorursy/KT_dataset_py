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
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "posts_answers" table
answers_table_ref = dataset_ref.table("posts_answers")

# API request - fetch the table
answers_table = client.get_table(answers_table_ref)

# Preview the first five lines of the "posts_answers" table
# client.list_rows(answers_table, max_results=5).to_dataframe()

# Construct a reference to the "posts_questions" table
questions_table_ref = dataset_ref.table("posts_questions")

# API request - fetch the table
questions_table = client.get_table(questions_table_ref)

# Preview the first five lines of the "posts_questions" table
# client.list_rows(questions_table, max_results=5).to_dataframe()

# Your code here
# bad_query is identical to working_query but has an extra whitespace after q in line 3!!!
bad_query = """
                SELECT a.id, a.body, a.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
                INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q 
                    ON q.id = a.parent_id
                WHERE q.tags LIKE '%bigquery%'
                """

working_query = """
                SELECT a.id, a.body, a.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_answers` AS a
                INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` AS q
                    ON q.id = a.parent_id
                WHERE q.tags LIKE '%bigquery%'
                """


# Note use_query_cache=False
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10, use_query_cache=False)
# Using *working_query*
answers_query_job = client.query(working_query, job_config=safe_config) # Your code goes here

# Returns an error, because cache is disabled. If you change to use_query_cache=True above,
# it will pass and below will print Cache hit: True
answers_results = answers_query_job.to_dataframe() # Your code goes here

# print("Cache hit: ")
# print(answers_query_job.cache_hit)

# # Preview results
# print(answers_results.head())
