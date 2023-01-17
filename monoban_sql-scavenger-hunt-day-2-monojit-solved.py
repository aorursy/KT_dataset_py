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
#print the first couple rows of the "full" table
hacker_news.head("full")

#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """
        SELECT type, count(id) 
        FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
"""

#saving to dataframe
how_many_stories_type = hacker_news.query_to_pandas_safe(query)
#shwoing them
how_many_stories_type.head()
#How many comments have been deleted? (If a comment was deleted the "deleted" column in the
#comments table will have the value "True".)
query = """
        SELECT count(id) 
        FROM `bigquery-public-data.hacker_news.comments`
        WHERE deleted = True
"""

deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
#Optional extra credit: read about aggregate functions other than COUNT()
#and modify one of the queries you wrote above to use a different aggregate function.

query = """
        SELECT countif(deleted=True)
        FROM `bigquery-public-data.hacker_news.comments`
"""

deleted_comments_opt = hacker_news.query_to_pandas_safe(query)
deleted_comments_opt.head()