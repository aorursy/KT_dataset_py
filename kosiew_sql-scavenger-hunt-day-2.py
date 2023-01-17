# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
table = "`bigquery-public-data.hacker_news.comments`"
query = f"""SELECT parent, COUNT(id)
            FROM {table}
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# print information on all the columns in the table
table = "comments"
hacker_news.table_schema(table)

table
# Your code goes here :)
table = "`bigquery-public-data.hacker_news.full`"
"""
* How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
"""
query = f"""
        select type,
          count(id) cnt
        from {table}
        group by type"""


hacker_news.estimate_query_size(query)
max_gb_scanned = 1
story_types = hacker_news.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
story_types
# print information on all the columns in the table
table = "comments"
hacker_news.table_schema(table)

table = "`bigquery-public-data.hacker_news.comments`"
bqh = hacker_news
query = f"""select deleted,
               count(0) cnt
           from {table}
           group by deleted"""
bqh.estimate_query_size(query)
"""
* How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
"""

max_gb_scanned = 1
comments_deleted_summary = bqh.query_to_pandas_safe(query, max_gb_scanned=max_gb_scanned)
comments_deleted_summary








"""
* How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
* How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
* **Optional extra credit**: read about [aggregate functions other than COUNT()](https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) and modify one of the queries you wrote above to use a different aggregate function.
"""