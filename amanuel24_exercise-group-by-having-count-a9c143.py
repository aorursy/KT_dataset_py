# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")
query = '''select type, count(id)
           from `bigquery-public-data.hacker_news.full`
           group by type'''
count_type = hacker_news.query_to_pandas_safe(query)
count_type
query = '''select deleted, count(id)
           from `bigquery-public-data.hacker_news.full`
           group by deleted
           having deleted = True'''
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments
query = '''select type, avg(score)
           from `bigquery-public-data.hacker_news.full`
           group by type'''
avg_score = hacker_news.query_to_pandas_safe(query)
avg_score