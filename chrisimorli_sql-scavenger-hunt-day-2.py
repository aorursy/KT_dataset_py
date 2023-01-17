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
hacker_news.head("full")

# How many stories (use the "id" column) are there of each type (in the "type" column)
# in the full table?
my_first_query = """SELECT type COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
popular_types = hacker_news.query_to_pandas_safe(my_first_query)
popular_types.head()
# How many comments have been deleted? (If a comment was deleted the "deleted" column
# in the comments table will have the value "True".)
my_second_query = """SELECT type, deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type, deleted
            HAVING deleted = True AND type = 'comment'
            """
# How many comments have been deleted? (If a comment was deleted the "deleted" column
# in the comments table will have the value "True".)

# The following does not work 
my_second_query = """SELECT type, deleted , COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
            """
# but this here and I don't know why
my_second_query = """SELECT COUNT(type), deleted
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
            """
deleted_comments = hacker_news.query_to_pandas_safe(my_second_query)
deleted_comments.head()
# latest entry of each type
my_third_query = """SELECT Max(time), type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
latest_entries = hacker_news.query_to_pandas_safe(my_third_query)
latest_entries.head()