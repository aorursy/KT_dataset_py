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
# Your code goes here :)
# query to pass to how many stories there are 
# of each type in the full table
query = """SELECT type, COUNT(title)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_count = hacker_news.query_to_pandas_safe(query)
type_count.head()
#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()

# Try aggregate from 
# https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions
query = """SELECT author, MAX(id)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            HAVING COUNT(id) > 100
        """
writes_alot = hacker_news.query_to_pandas_safe(query)
writes_alot.head()