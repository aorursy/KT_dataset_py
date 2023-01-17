# import package with helper functions 
import bq_helper
import pandas as pd

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.table_schema("comments")
# column 'type' no longer exists in table, used 'author' instead
query = """SELECT author, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author"""
types = hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.2)
types.head(10)
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
hacker_news.estimate_query_size(query)
deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments)
# get the average posting time per author
query = """SELECT author, AVG(time)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author"""
posting_time = hacker_news.query_to_pandas_safe(query)
posting_time["datetime"] = pd.to_datetime(posting_time["f0_"], unit="s")
posting_time.head(10)