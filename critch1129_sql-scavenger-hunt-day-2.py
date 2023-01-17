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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")

query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_type = hacker_news.query_to_pandas_safe(query)

stories_type.head()
# There are 2845239 stories
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
hacker_news.head("comments")

query_comm_del = """ SELECT deleted, COUNT(id)
                     FROM `bigquery-public-data.hacker_news.comments`
                     GROUP BY deleted
                 """
comment_del = hacker_news.query_to_pandas_safe(query_comm_del)

comment_del.head()
# 227736 comments have been deleted

# **Optional extra credit**: read about [aggregate functions other than COUNT()]
# and modify one of the queries you wrote above to use a different aggregate function.

hacker_news.head("full")

query_xtra = """ SELECT type, SUM(id)
                 FROM `bigquery-public-data.hacker_news.full`
                 GROUP BY type
             """
xtra_credit = hacker_news.query_to_pandas_safe(query_xtra)

xtra_credit.head()
# Here I used the sum of id's by type, which may be non-sensical, but it works!