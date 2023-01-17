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
# How many stories (use the "id" column) are there of each type (in the "type" column) 
# in the full table?
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_type = hacker_news.query_to_pandas_safe(query1)
story_type.head()

# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query2 = """SELECT deleted, COUNT(type)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
         """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head(1)

# read about [aggregate functions other than COUNT()] 
# and modify one of the queries you wrote above to use a different aggregate function.
query3 = """SELECT deleted, SUM(time)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
         """
deleted_comments_time = hacker_news.query_to_pandas_safe(query3)
deleted_comments_time.head()
