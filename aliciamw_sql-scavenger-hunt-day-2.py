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
# print the first couple rows of the "full" table
hacker_news.head("full")
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_type = hacker_news.query_to_pandas_safe(query1)
stories_type
# How many comments have been deleted?
query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted
        """
comments_deleted = hacker_news.query_to_pandas_safe(query2)
comments_deleted
# How many comments have been deleted?
query3 = """SELECT COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
        """
comments_deleted2 = hacker_news.query_to_pandas_safe(query3)
comments_deleted2