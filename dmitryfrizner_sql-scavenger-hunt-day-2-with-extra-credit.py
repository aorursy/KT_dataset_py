# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="hacker_news"
)
hacker_news.head("full")
query1 = """
    SELECT type, COUNT(id) AS total_by_type
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    ORDER BY total_by_type DESC
"""
result1 = hacker_news.query_to_pandas_safe(query1)
result1
query2 = """
    SELECT COUNT(id) AS DELETED_COMMENTS
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
"""
result2 = hacker_news.query_to_pandas_safe(query2)
result2[['DELETED_COMMENTS']]
query2_mod1 = """
    SELECT SUM(CAST(deleted as INT64)) AS DELETED_COMMENTS
    FROM `bigquery-public-data.hacker_news.comments`
"""
result2_mod1 = hacker_news.query_to_pandas_safe(query2_mod1)
result2_mod1[['DELETED_COMMENTS']]