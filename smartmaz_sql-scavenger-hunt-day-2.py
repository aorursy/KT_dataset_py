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
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
type_id_query = """SELECT type, count(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""

type_id_set = hacker_news.query_to_pandas_safe(type_id_query)
type_id_set.head()



# How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# Optional extra credit**: 
#read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 
#and modify one of the queries you wrote above to use a different aggregate function.

count_deleted_comments_query = """SELECT COUNT(id), MIN(parent), MAX(parent)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = True
"""
hacker_news.estimate_query_size(count_deleted_comments_query)
hacker_news.query_to_pandas_safe(count_deleted_comments_query)


