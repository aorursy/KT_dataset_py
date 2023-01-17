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
# print the first couple rows of the "comments" table
hacker_news.head("full")
# 1) How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

stories_type_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_of_each_type = hacker_news.query_to_pandas_safe(stories_type_query)
stories_of_each_type.head()
# 2) How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

deleted_comments_query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(deleted_comments_query)
deleted_comments.head()
# 3) Modify one of the queries you wrote above to use a different aggregate function.

deleted_comments_query_if = """SELECT deleted, COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments_diff = hacker_news.query_to_pandas_safe(deleted_comments_query_if)
deleted_comments_diff.head()