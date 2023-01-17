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
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

query = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 0
        """

stories = hacker_news.query_to_pandas_safe(query)
stories.head()
# * How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
# **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
# Here I incorporate the condition WHERE deleted = True using the aggregate function COUNTIF
query = """SELECT COUNTIF(deleted) as count
            FROM `bigquery-public-data.hacker_news.comments`
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments