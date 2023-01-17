# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq # big_query helper package

# create helper object for bigQuery dataset
h_news= bq.BigQueryHelper(active_project= "bigquery-public-data", 
                          dataset_name = "hacker_news")

h_news.list_tables()
# printing all columns and it's metadata in 'comments' table
h_news.table_schema("comments")
h_news.head("comments")
h_news.head("comments", selected_columns="by", 
            num_rows=10)
# bigData datasets are verrrry big.
# So it's always better to estimate the query size without running the actual queries.
# Isn't it cool you can do that? I wonder HOW?

query = """select score from `bigquery-public-data.hacker_news.full` 
where type = "job" """

# check how big this query will be
h_news.estimate_query_size(query) # return in GB
# only run this query if it's less than 100 MB
h_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = h_news.query_to_pandas_safe(query)

job_post_scores.tail()
# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")

