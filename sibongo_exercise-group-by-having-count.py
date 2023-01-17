# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query = """
        SELECT type, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
"""
type_count = hacker_news.query_to_pandas_safe(query)
type_count.head()
# Your Code Here
query2 = """SELECT deleted, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
"""
print(hacker_news.estimate_query_size(query2))
deleted_count = hacker_news.query_to_pandas_safe(query2)
deleted_count.head()
## Now, let's verify that it's wokred by seeing if replacing = True with IS NULL results in a different query result
query3 = """SELECT deleted, COUNT(id) AS count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS NULL
            GROUP BY deleted
"""
print(hacker_news.estimate_query_size(query3))
not_deleted_count = hacker_news.query_to_pandas_safe(query3)
not_deleted_count.head()
# It works!
# Your Code Here
query4 = """
        SELECT AVG(SUBQUERY.count) as average
        FROM (SELECT author, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author) AS SUBQUERY
"""
print(hacker_news.estimate_query_size(query4))
avg = hacker_news.query_to_pandas_safe(query4)
avg