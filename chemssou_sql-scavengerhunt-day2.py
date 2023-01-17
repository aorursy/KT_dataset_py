import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
# import package with helper functions
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the fisrst couple rows of the "comments" table
hacker_news.head("comments")
query = """ SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
            """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.list_tables()
hacker_news.head("full")
hacker_news.head("stories")
query2 = """ SELECT type, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
             HAVING type = "story" 
             """
number_stories = hacker_news.query_to_pandas_safe(query2)
number_stories.head()
hacker_news.head("comments")
query3 = """ SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING deleted = True
             """
deleted_comments = hacker_news.query_to_pandas_safe(query3)
deleted_comments.head()
query4 = """ SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             WHERE deleted = True
             GROUP BY deleted
             """
deleted_comments2 = hacker_news.query_to_pandas_safe(query4)
deleted_comments2.head()
