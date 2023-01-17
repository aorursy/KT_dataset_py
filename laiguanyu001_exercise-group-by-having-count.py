# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query = """SELECT type,COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type"""
output = hacker_news.query_to_pandas_safe(query)
print(output.head())
# Your Code Here
query = """select deleted,count(id)
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted"""
output = hacker_news.query_to_pandas(query)
print(output.head())
# Your Code Here
query = """SELECT AVG(time) as avg
        FROM `bigquery-public-data.hacker_news.comments`"""
output = hacker_news.query_to_pandas(query)
print(output.head())
#taking average of time from table 'comments'
query = """SELECT FORMAT("%T", ARRAY_AGG(DISTINCT dead)) AS array_agg
FROM `bigquery-public-data.hacker_news.comments`
"""
output = hacker_news.query_to_pandas(query)
print(output.head())
#output all input in column 'dead' as an array