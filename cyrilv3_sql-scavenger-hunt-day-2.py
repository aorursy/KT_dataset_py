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
popular_stories.head()
hacker_news.head('full')
# Your code goes here :)
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_count = hacker_news.query_to_pandas_safe(query)
type_count.head()
hacker_news.head("comments")
# Your code goes here :)
# query to pass to 
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
# Your code goes here :)
# query to pass to 
# this query gives you the number of stories where parent id was greater than comment id
# grouped by their type
query = """SELECT type, COUNTIF(parent > id) as pvc
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING pvc > 1
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
parent_v_child_type= hacker_news.query_to_pandas_safe(query)
parent_v_child_type.head()