# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# import package with helper functions 
import bq_helper
import pandas as pd
import numpy as np


# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
print("Tables list:")
print(hacker_news.list_tables())
# print the first couple rows of the "comments" table
hacker_news.head("full")
# query to pass to for question 1:
query_type_cnts = """SELECT type, COUNT(id) as type_counts
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(ID) desc
        """
# cost of query
print("Query size estimate:")
print(hacker_news.estimate_query_size(query_type_cnts))
story_type_cnts = hacker_news.query_to_pandas_safe(query_type_cnts)
print(story_type_cnts)
print(hacker_news.head("comments"))
# query to pass to for question 2:
query_del_com = """SELECT COUNT(deleted) as comments_deleted
            FROM `bigquery-public-data.hacker_news.comments`
            group by deleted
            having deleted = true
            """

# cost of query
print("Query size estimate:")
print(hacker_news.estimate_query_size(query_del_com))
# running query for question 2:
del_comments_cnts = hacker_news.query_to_pandas_safe(query_del_com)
print(del_comments_cnts)
query_del_com_countif = """SELECT COUNTIF(deleted=True) as comments_deleted_cntif
            FROM `bigquery-public-data.hacker_news.comments`
                       """
# running query for optional question:
del_comments_countif = hacker_news.query_to_pandas_safe(query_del_com_countif)
print(del_comments_countif)
#Cheating with the sum function

query_del_com_sum = """SELECT SUM(1) as comments_deleted_sum
            FROM `bigquery-public-data.hacker_news.comments`
            group by deleted
            having deleted = true
            """
del_comments_sum = hacker_news.query_to_pandas_safe(query_del_com_sum)
print(del_comments_sum)
