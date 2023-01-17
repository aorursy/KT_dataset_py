# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
query = """SELECT count(id), type
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type"""

cl = hacker_news.query_to_pandas(query)
cl.head()

query2 = """SELECT count(id), deleted
FROM `bigquery-public-data.hacker_news.full`
GROUP BY deleted
HAVING deleted = True """

app = hacker_news.query_to_pandas(query2)
app.head()
query3 = """SELECT min(id),deleted
FROM `bigquery-public-data.hacker_news.full`
GROUP BY deleted"""

cg = hacker_news.query_to_pandas_safe(query3)
cg.head()