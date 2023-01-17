# import package with helper functions 
import bq_helper
import pandas as pd
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
# Your code goes here :)

# QUESTION 1
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# print the first couple rows of the full dataset
hacker_news.head("full")

# query to pass to 
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_types = hacker_news.query_to_pandas_safe(query1)

story_types.head()
# QUESTOIN 2
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# query to pass to 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query2)

deleted_comments.head()
# QUESTION 3
# **Optional extra credit**: read about aggregate functions other than COUNT()
# and modify one of the queries you wrote above to use a different aggregate function.
# (https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 

# Which type story is the least frequently published? 

query3 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(id)
        """

min_type = hacker_news.query_to_pandas_safe(query3)

min_type.head(1)