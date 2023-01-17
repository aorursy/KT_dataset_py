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
# display multiple print results on one line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# how many stories (use the id column) are there of each type 

query_story = """SELECT type, COUNT(id) AS total_type_count
                  FROM `bigquery-public-data.hacker_news.full`
                  GROUP BY type
              """
print(query_story)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_type = hacker_news.query_to_pandas_safe(query_story)
# prints stories of each type f0_ is equal to the count 
story_type
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table 
# will have the value "True".)

query_delete = """SELECT COUNT(deleted) as deleted_count
                  FROM `bigquery-public-data.hacker_news.comments`
                  GROUP BY deleted
                  HAVING deleted = True
              """
print(query_delete)
# print in pandas df
deleted_comments = hacker_news.query_to_pandas_safe(query_delete)
# prints deleted comments count
deleted_comments
# Optional extra credit**: read about 
# [aggregate functions other than COUNT()]
# and modify one of the queries you wrote above to use a different aggregate function.

min_id = """SELECT type, MIN(id) AS min_id
                  FROM `bigquery-public-data.hacker_news.full`
                  GROUP BY type
              """
print(min_id)
# print in pandas df
ID_min = hacker_news.query_to_pandas_safe(min_id)
ID_min