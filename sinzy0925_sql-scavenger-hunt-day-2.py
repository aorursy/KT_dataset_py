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
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

#query = """SELECT id, type
#            FROM `bigquery-public-data.hacker_news.full`
#        """
#popular_stories = hacker_news.query_to_pandas_safe(query)
#popular_stories.head()
#open_aq.head("global_air_quality",100)
hacker_news.head("full",100)
hacker_news.head("comments",100)
# Your code goes here :)
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

query = """SELECT type,count(id)
            FROM `bigquery-public-data.hacker_news.full`
            group by type
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()

# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query = """SELECT count(deleted) as deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories#.head()

# **Optional extra credit**: read about [aggregate functions other than COUNT()]
# (https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 
# and modify one of the queries you wrote above to use a different aggregate function.

query = """SELECT COUNTIF(deleted = True) as deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            --where deleted = True
            --group by author
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories#.head()
