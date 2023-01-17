# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.list_tables()
hacker_news.head("full")
hacker_news.head("comments")
query_full = """SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type"""
hacker_news.estimate_query_size(query_full) * 1000
story_count = hacker_news.query_to_pandas_safe(query_full)
print(story_count)
print(story_count.query("type == 'story'"))
query_deleted = """SELECT deleted, COUNT(id)
                   FROM `bigquery-public-data.hacker_news.comments`
                   GROUP BY deleted"""
hacker_news.estimate_query_size(query_deleted) * 1000
deleted_comments_count = hacker_news.query_to_pandas_safe(query_deleted)
print(deleted_comments_count)
print(deleted_comments_count.query("deleted == True"))
query_max_score = """SELECT type, MAX(score)
                     FROM `bigquery-public-data.hacker_news.full`
                     GROUP BY type"""
hacker_news.estimate_query_size(query_max_score) * 1000
max_score = hacker_news.query_to_pandas_safe(query_max_score)
print(max_score)