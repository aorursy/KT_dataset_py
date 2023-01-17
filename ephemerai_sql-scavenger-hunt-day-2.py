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

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# How many stories are there of each type in the full table?

# print the first couple rows of the "full" table
hacker_news.head("full")

query_1 = """
            SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
story_type = hacker_news.query_to_pandas_safe(query_1)
story_type.head()
# How many comments have been deleted? 
query_2 = """
            SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
deleted_comment = hacker_news.query_to_pandas_safe(query_2)
deleted_comment.head()
# Or:
# How many comments have been deleted? 
query_2 = """
            SELECT deleted, COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
            """
deleted_comment = hacker_news.query_to_pandas_safe(query_2)
deleted_comment.head()
# Another way: 
# How many comments have been deleted? 
query_2 = """
            SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            """
deleted_comment = hacker_news.query_to_pandas_safe(query_2)
deleted_comment.head()
# Optional extra credit: use a different aggregate function
# What is the average number of descendants of each type? 
query_3 = """
            SELECT type, AVG(descendants)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
type_descendants = hacker_news.query_to_pandas_safe(query_3)
type_descendants.head()