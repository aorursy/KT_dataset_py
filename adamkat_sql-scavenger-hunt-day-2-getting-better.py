# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
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
query1 = """
SELECT
    type,
    COUNT(id) count
FROM
    `bigquery-public-data.hacker_news.full`
GROUP BY type
"""

stories_by_type = hacker_news.query_to_pandas_safe(query1)
stories_by_type
query2 = """
SELECT
    COUNT(id) comments_deleted
FROM
    `bigquery-public-data.hacker_news.full`
WHERE
    deleted = True
"""

comments_deleted = hacker_news.query_to_pandas_safe(query2)
comments_deleted
query3 = """
SELECT
    MAX(descendants) max_descendants
FROM
    `bigquery-public-data.hacker_news.full`
"""

max_descendants = hacker_news.query_to_pandas_safe(query3)
max_descendants