# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.list_tables()
hacker_news.head("comments")
hacker_news.head("full")
hacker_news.head("stories")
query_1 = """SELECT type, COUNT(id)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY type
          """
no_of_stories = hacker_news.query_to_pandas_safe(query_1)
print(no_of_stories.head())
query_2 = """SELECT deleted, COUNT(deleted)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY deleted
          HAVING deleted = True
          """
deleted_comments = hacker_news.query_to_pandas_safe(query_2)
print(deleted_comments)
          
query_3 = """SELECT type, SUM(id)
          FROM `bigquery-public-data.hacker_news.full`
          GROUP BY type
          """
sum_of_stories = hacker_news.query_to_pandas_safe(query_3)
print(sum_of_stories.head())