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
hacker_news.head("full").head()
# Your code goes here :)
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head()
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
# Note that GROUP BY is not needed for this query
query = """SELECT COUNT(*)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """

true_deleted_comments = hacker_news.query_to_pandas_safe(query)
true_deleted_comments.head()
# Check performance
query_with_groupby = """
    SELECT deleted, COUNT(id)
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY deleted
"""

query_without_groupby = """
    SELECT COUNT(*)
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
"""

%timeit _ = hacker_news.query_to_pandas_safe(query_with_groupby)
%timeit _ = hacker_news.query_to_pandas_safe(query_without_groupby)