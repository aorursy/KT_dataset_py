# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query = """SELECT type, COUNT(id) AS number
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
types = hacker_news.query_to_pandas_safe(query)
types.head()
query = """SELECT deleted, COUNT(id) AS number
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
HAVING deleted = True
"""
deleted = hacker_news.query_to_pandas_safe(query)
deleted.head()
query = """SELECT COUNTIF(deleted = True) AS number
FROM `bigquery-public-data.hacker_news.comments`
"""
deleted = hacker_news.query_to_pandas_safe(query)
deleted.head()