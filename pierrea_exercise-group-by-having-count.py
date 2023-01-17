# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("full")
query = """SELECT type, COUNT(id) AS NumStories

            FROM `bigquery-public-data.hacker_news.full`

            GROUP BY type

        """



stories = hacker_news.query_to_pandas_safe(query)
stories
query = """SELECT COUNT(id) AS NumDeletedComs

            FROM `bigquery-public-data.hacker_news.full`

            WHERE deleted = True

        """



deleted_count = hacker_news.query_to_pandas_safe(query)

deleted_count
query = """SELECT MAX(score) AS max_score

            FROM `bigquery-public-data.hacker_news.full`

            WHERE score >= 0

        """



avg_scores = hacker_news.query_to_pandas_safe(query)

avg_scores