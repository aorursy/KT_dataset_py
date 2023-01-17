import bq_helper as bq
hacker_news = bq.BigQueryHelper(active_project="bigquery-public-data",
                               dataset_name="hacker_news")
query = """SELECT COUNT(id) AS unique_stories
           FROM `bigquery-public-data.hacker_news.stories`
"""
hacker_news.estimate_query_size(query)
unique_stories_count = hacker_news.query_to_pandas_safe(query)
unique_stories_count
query2 = """SELECT COUNT(deleted) AS deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
"""
hacker_news.estimate_query_size(query2)
deleted_comments_count = hacker_news.query_to_pandas_safe(query2)
deleted_comments_count
query3 = """SELECT MIN(score) AS min_score,
                   AVG(score) AS average_score,
                   MAX(score) AS max_score,
                   type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY type ASC
"""
hacker_news.estimate_query_size(query3)
summary_scores = hacker_news.query_to_pandas_safe(query3)
summary_scores