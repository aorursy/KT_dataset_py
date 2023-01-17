# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_counts = hacker_news.query_to_pandas_safe(query)
print(type_counts)
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments)
query = """SELECT COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
        """
deleted_comments_number = hacker_news.query_to_pandas_safe(query)
print(deleted_comments_number)