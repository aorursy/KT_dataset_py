# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
print(hacker_news.list_tables())
# print the first couple rows of the "comments" table
hacker_news.head("comments")
df=hacker_news.query_to_pandas_safe("""
SELECT COUNT(id), type FROM `bigquery-public-data.hacker_news.full` GROUP BY type
""")
df
df=hacker_news.query_to_pandas_safe("""
SELECT COUNT(id) FROM `bigquery-public-data.hacker_news.full` WHERE deleted=True
""")
df
df=hacker_news.query_to_pandas_safe("""
SELECT AVG(score) FROM `bigquery-public-data.hacker_news.full` WHERE deleted=False AND score IS NOT NULL
""")
df