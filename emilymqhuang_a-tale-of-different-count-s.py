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
# My code goes here :)
# 1. How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# check the first couple rows of the "full" table
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_count = hacker_news.query_to_pandas_safe(query1)
type_count
# 2. How many comments have been deleted?

query2 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is True           
        """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments

# Checking the size of query1:
hacker_news.estimate_query_size(query1)
query1_2 = """SELECT type, COUNT(*)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_count_2 = hacker_news.query_to_pandas_safe(query1_2)
type_count_2
# Check the size of query1_2:
hacker_news.estimate_query_size(query1_2)
# Check the size of query2:
hacker_news.estimate_query_size(query2)
query2_2 = """SELECT COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is True           
        """
deleted_comments_2 = hacker_news.query_to_pandas_safe(query2_2)
deleted_comments_2
# Check the size of query2_2:
hacker_news.estimate_query_size(query2_2)