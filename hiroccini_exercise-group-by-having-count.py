# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
print(f"hacker_news.list_tables(): {hacker_news.list_tables()}")
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
hacker_news.head("full")
query = """SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type"""
types = hacker_news.query_to_pandas_safe(query)
types
hacker_news.head("comments")
query = """SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted"""
deleted_c = hacker_news.query_to_pandas_safe(query)
print(f"{deleted_c}")
t_row = deleted_c['deleted'] == True
deleted_c.loc[t_row, 'f0_']
for i, v in enumerate(deleted_c['f0_']):
    print(f"{i}: {v}, {deleted_c['deleted'][i]}")
answer = [v for i, v in enumerate(deleted_c['f0_']) if deleted_c['deleted'][i] == True][0]
print(f"answer: {answer}")
hacker_news.head("full")
# Your Code Here
query = """SELECT type, COUNT(id), min(score), max(score)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type"""
modified = hacker_news.query_to_pandas_safe(query)
modified
