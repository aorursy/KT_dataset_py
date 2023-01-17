# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.table_schema("comments")
# hacker_news.head("comments")
# hacker_news.list_tables()
# hacker_news.head("full")
# hacker_news.head("full_201510")
# hacker_news.head("stories")
# Your Code Here
query = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
"""
story = hacker_news.query_to_pandas_safe(query)
story.head()
# Your Code Here
query = """SELECT COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           WHERE deleted = True
"""
dele = hacker_news.query_to_pandas_safe(query)
print(dele)
# Your Code Here
#in these docs 文件打不开
