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
QUERY = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""
stories = hacker_news.query_to_pandas_safe(QUERY)


stories.head()
QUERY = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True """
num_deleted = hacker_news.query_to_pandas_safe(QUERY, max_gb_scanned=2)
num_deleted
QUERY = """SELECT MIN(time)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "comment" """
earliest_comment = hacker_news.query_to_pandas_safe(QUERY)
earliest_comment
