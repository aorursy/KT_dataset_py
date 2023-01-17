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
#How many stories are there of each type in the table called full 
query1 = """SELECT type, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """
#check how big this query will be
hacker_news.estimate_query_size(query1)
#query returns a dataframe only if query is smaller than 0.1GB
full_types = hacker_news.query_to_pandas_safe(query1)
full_types
#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)-bool
query2 = """SELECT COUNT(deleted), deleted
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted
        HAVING deleted = True"""
        
#check how big this query will be
hacker_news.estimate_query_size(query2)
#query returns a dataframe only if query is smaller than 0.1GB
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments