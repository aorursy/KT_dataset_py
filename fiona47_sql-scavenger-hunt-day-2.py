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

stories_query = """SELECT type, COUNT(id) as Count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_count = hacker_news.query_to_pandas_safe(stories_query)

stories_count
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

hacker_news.head("comments")

true_count_query =  """SELECT count(deleted) as True_Count
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """

true_count = hacker_news.query_to_pandas_safe(true_count_query)

true_count
# Optional extra credit: read about [aggregate functions other than COUNT()] and modify one of the queries you wrote above to use a different aggregate function.

true_count_query =  """SELECT countif(deleted = True) as True_Count
            FROM `bigquery-public-data.hacker_news.comments`
        """

true_count = hacker_news.query_to_pandas_safe(true_count_query)

true_count