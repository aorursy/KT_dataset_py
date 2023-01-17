import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
Query_1 = """SELECT Type, COUNT(id) as ID_Count, AVG(score) as Average_score
FROM `bigquery-public-data.hacker_news.full`
GROUP BY Type
"""
Query_2 = """SELECT COUNT(id) as QTY_of_deleted_comments
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = True
"""
Story_type = hacker_news.query_to_pandas_safe(Query_1)
Story_type.head
deleted_count = hacker_news.query_to_pandas_safe(Query_2)
deleted_count.head
