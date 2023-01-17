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
hacker_news.table_schema('full')
hacker_news.head('full')
# Your code goes here 
types_query = """SELECT type, COUNT(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY type"""
 

different_types = hacker_news.query_to_pandas_safe(types_query)
different_types
deleted_query ="""SELECT deleted, COUNT(id) FROM `bigquery-public-data.hacker_news.full` GROUP BY deleted HAVING deleted = True""" 
hacker_news.query_to_pandas_safe(deleted_query)
types_score = """SELECT type, MAX(score), AVG(score) FROM `bigquery-public-data.hacker_news.full`  GROUP BY type"""
Max_Avg_score_types = hacker_news.query_to_pandas_safe(types_score)
# Maximum and Avg story score by type 
Max_Avg_score_types

