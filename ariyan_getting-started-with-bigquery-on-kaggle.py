# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# the custom helper package for BigQuery
import bq_helper
#helper object for the bigQuery dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")
hacker_news.list_tables()
#table_schema method takes the table name as input.
hacker_news.table_schema('full')
hacker_news.head("full")
hacker_news.head('full', selected_columns='by', num_rows=10)
#this is the formal query similar to one you run on SQL 

query = '''SELECT score 
           FROM `bigquery-public-data.hacker_news.full`
           WHERE type="job" '''
# for every query test the size of the query before running the actual query.

hacker_news.estimate_query_size(query)
#Actually running a query
#there are two options for that 
#1. BigQueryHelper.query_to_pandas(query): This method takes a query and returns a Pandas dataframe.
#2. BigQueryHelper.query_to_pandas_safe(query, max_gb_scanned=1): This method takes a query and returns a Pandas dataframe only if the size of the query is less than the upperSizeLimit (1 gigabyte by default).

#example with safe query
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
job_post_score = hacker_news.query_to_pandas_safe(query)#default safe size is 1 GB
#the pandas dataframe are returned by bigQuery can be treated like any other pandas dataframe.
job_post_score.score.mean()
#save the data in .csv
job_post_score.to_csv("job_post_score.csv")
