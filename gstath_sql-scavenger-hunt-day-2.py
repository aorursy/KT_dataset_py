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
# print all the tables in this dataset (there's only one!)
hacker_news.list_tables()
# Your code goes here :)

# query to select 
# How many stories (use the "id" column) are there of each type 
# (in the "type" column) in the full table?
query_stories_per_type = """SELECT type, count(id) as `count`
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """

stories_per_type = hacker_news.query_to_pandas_safe(query_stories_per_type)

stories_per_type
# How many comments have been deleted? 

query_deleted_comments = """SELECT count(id) as `deleted_comments`
                   FROM `bigquery-public-data.hacker_news.comments`
                   WHERE deleted = true
                   """
deleted_comments = hacker_news.query_to_pandas_safe(query_deleted_comments)

deleted_comments
# other aggregate functions.
query_other = """SELECT type, count(id) as `total_count`,
            AVG(score) as `avg_score`,
            MIN(score) as `min_score`,
            MAX(score) as `max_score`
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """

other = hacker_news.query_to_pandas_safe(query_other)

other

