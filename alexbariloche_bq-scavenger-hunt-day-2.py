import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# import our bq_helper package
import bq_helper

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
query = """SELECT COUNT(*) AS unique_ids
           FROM (
               SELECT id, COUNT(*)
               FROM `bigquery-public-data.hacker_news.full`
               GROUP BY id)"""
hacker_news.estimate_query_size(query)
unique_ids = hacker_news.query_to_pandas_safe(query)
unique_ids.head()
query = """SELECT COUNT(*) AS deleted_comments
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True"""
hacker_news.estimate_query_size(query)
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()