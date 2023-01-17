# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
"""
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head()

query_2 = """ SELECT COUNT(id)
              FROM `bigquery-public-data.hacker_news.comments`
              WHERE deleted = True
"""

deleted_comments = hacker_news.query_to_pandas_safe(query_2)
deleted_comments.head()
