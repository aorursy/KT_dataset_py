# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.list_tables()
# query to pass to 
query = """SELECT parent, COUNT(id) AS comment_count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10"""
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Your code goes here :)
hacker_news.head("full")
type_query = """SELECT type, COUNT(id) as num_stories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""
hacker_news.estimate_query_size(type_query)
num_type_stories = hacker_news.query_to_pandas_safe(type_query)
num_type_stories.head()
deletes_query = """SELECT count(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    WHERE deleted = True AND type = "comment" """
hacker_news.estimate_query_size(deletes_query)
deleted_comments = hacker_news.query_to_pandas_safe(deletes_query)
deleted_comments.head()
