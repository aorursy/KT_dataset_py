import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query = """ SELECT type, COUNT(*) AS count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
hacker_news.estimate_query_size(query)
type_count = hacker_news.query_to_pandas_safe(query)
type_count
# extra exercise:
# calculate the percentage of each type of story
query = """ SELECT type, cnt AS count, cnt/total*100 AS percentage
                    FROM ( SELECT COUNT(*) as total
                                 FROM `bigquery-public-data.hacker_news.full` ) c,
                               ( SELECT type, COUNT(*) as cnt
                                 FROM `bigquery-public-data.hacker_news.full`
                                 GROUP BY type ) d
                """
hacker_news.estimate_query_size(query)
type_count_avg = hacker_news.query_to_pandas_safe(query)
type_count_avg
query = """ SELECT deleted, COUNT(id) AS count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY deleted
                HAVING deleted = True
                """
hacker_news.estimate_query_size(query)
deleted_comment = hacker_news.query_to_pandas_safe(query)
deleted_comment
query = """ SELECT type, COUNTIF(deleted=True) AS count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
hacker_news.estimate_query_size(query)
type_deleted_count = hacker_news.query_to_pandas_safe(query)
type_deleted_count
