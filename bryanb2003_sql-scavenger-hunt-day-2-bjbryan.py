# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query to pass to 
query = """SELECT type, COUNT(distinct id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

story_types = hacker_news.query_to_pandas_safe(query)

print(story_types)
query = """SELECT deleted, COUNT(distinct id)
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted = True
            GROUP BY deleted
        """
deleted_coms = hacker_news.query_to_pandas_safe(query)

print(deleted_coms)
query = """SELECT type, AVG(score) as AVG, COUNT(distinct id) as CNT
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
score_types = hacker_news.query_to_pandas_safe(query)

print(score_types)