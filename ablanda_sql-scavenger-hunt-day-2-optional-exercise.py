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
#  COUNT not allowed in WHERE clause
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Exercise 1
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.estimate_query_size(query1)
type_stories = hacker_news.query_to_pandas_safe(query1)
type_stories
# Exercise 2
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
hacker_news.estimate_query_size(query2)
deleted_stories = hacker_news.query_to_pandas_safe(query2)
type_stories
# Exercise 3
#ANYVALUE
query3 = """SELECT ANY_VALUE(id) as any_value
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
hacker_news.estimate_query_size(query3)
deleted_stories = hacker_news.query_to_pandas_safe(query3)
deleted_stories

# AVG
query3 = """SELECT AVG(time) as avg
            FROM `bigquery-public-data.hacker_news.comments`
        """
hacker_news.estimate_query_size(query3)
deleted_stories = hacker_news.query_to_pandas_safe(query3)
deleted_stories
# COUNTIF
query3 = """SELECT COUNTIF(deleted=True) AS not_null
            FROM `bigquery-public-data.hacker_news.comments`
        """
hacker_news.estimate_query_size(query3)
deleted_stories = hacker_news.query_to_pandas_safe(query3)
deleted_stories
# STRING_AGG
query3 = """SELECT STRING_AGG(author, " & " LIMIT 2) AS string_agg
            FROM `bigquery-public-data.hacker_news.comments`
        """
hacker_news.estimate_query_size(query3)
deleted_stories = hacker_news.query_to_pandas_safe(query3)
deleted_stories