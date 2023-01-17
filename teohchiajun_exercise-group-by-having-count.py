# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# print the first few rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories)
# Your Code Here
query_2 = """SELECT COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted = True
          """
deleted_com = hacker_news.query_to_pandas_safe(query_2)
print(deleted_com)
# Your Code Here
query_3 = """SELECT max(timestamp)
              FROM `bigquery-public-data.hacker_news.full`
              WHERE deleted = True
          """
latest_del = hacker_news.query_to_pandas_safe(query_3)
print(latest_del)