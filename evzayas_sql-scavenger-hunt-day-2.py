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
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head(20)
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted = hacker_news.query_to_pandas_safe(query)
deleted
# * **Optional extra credit**: read about [aggregate functions other than COUNT()]
# and modify one of the queries you wrote above to use a different aggregate function.

# what are the range of scores of the stories, and what is the average score?

query = """SELECT DISTINCT score
            FROM `bigquery-public-data.hacker_news.stories`
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
scores = hacker_news.query_to_pandas_safe(query)
scores.count()
query = """SELECT AVG(score)
            FROM `bigquery-public-data.hacker_news.stories`
            WHERE score > 0
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
scores_avg = hacker_news.query_to_pandas_safe(query)
scores_avg
