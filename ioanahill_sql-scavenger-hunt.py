# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()

# print information on all the columns in the "full" table in the hacker_news dataset
hacker_news.table_schema("full")
hacker_news.head('full',selected_columns='by', num_rows=10)
# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)

# 0.151 = 150 MB (answer is in GB)
# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.score.mean()
# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")
