# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head("full")
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_counts = hacker_news.query_to_pandas_safe(query)
type_counts
hacker_news.head("comments")
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted = hacker_news.query_to_pandas_safe(query)
deleted
query = """SELECT MAX(descendants), type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP by type
        """
desc = hacker_news.query_to_pandas_safe(query)
desc