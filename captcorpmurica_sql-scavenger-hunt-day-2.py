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
# How many stories (use the "id" column) are there for each type in the full table?
d2q1 = """SELECT type, COUNT(id) AS freq
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_by_type = hacker_news.query_to_pandas_safe(d2q1)
stories_by_type
# How many comments have been deleted? (If a comment was deleted, the "deleted" column in the comments table will have the value "True".)
d2q2 = """SELECT COUNT(id) AS freq
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
number_deleted = hacker_news.query_to_pandas_safe(d2q2)
number_deleted
# Modify one of the queries you wrote above to use a different aggregate function.
d2q3 = """SELECT COUNTIF(deleted = True) AS freq
            FROM `bigquery-public-data.hacker_news.comments`
        """
countif_deleted = hacker_news.query_to_pandas_safe(d2q3)
countif_deleted