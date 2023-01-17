# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query21 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
hacker_news.query_to_pandas_safe(query21)
# Your Code Here
query22 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted """
hacker_news.query_to_pandas_safe(query22)
# Your Code Here
query23 = """SELECT COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.full`"""
hacker_news.query_to_pandas_safe(query23)