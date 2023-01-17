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
hacker_news.head("full")

query1 = """SELECT type,COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
types = hacker_news.query_to_pandas_safe(query1)
types.head()
# query to pass to 
# Query for counting deleted comments
query2 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()

query3 = """SELECT type, COUNT(deleted) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type HAVING MIN(score) > 0 """
groupcon_comments = hacker_news.query_to_pandas_safe(query3)
groupcon_comments.head()
query4 = """SELECT by, COUNT(type) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type HAVING MIN(score) > 0 """
groupcon_comments = hacker_news.query_to_pandas_safe(query4)
groupcon_comments.head()