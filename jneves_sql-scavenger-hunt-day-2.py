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
first_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_types = hacker_news.query_to_pandas_safe(first_query)
count_types.head()
second_query = """SELECT COUNT(id) AS deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(second_query)
deleted_comments.head()
optional_query = """SELECT type, MAX(score) AS score_max
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
max_score = hacker_news.query_to_pandas_safe(optional_query)
max_score.head()