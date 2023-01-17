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
query = """SELECT type, COUNT(id) as Amount
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 0
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories
query = """SELECT deleted,COUNT(id) as Amount
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """
deleted_stories = hacker_news.query_to_pandas_safe(query)
deleted_stories
query = """SELECT deleted,count(*) 
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY deleted
        """

deleted_stories = hacker_news.query_to_pandas_safe(query)
deleted_stories