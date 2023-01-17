# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_hn = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

open_hn.list_tables()

open_hn.head("full")
# query to SELECT and COUNT all the DISTINCT ids
query = """SELECT type, COUNT(DISTINCT id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
unique_stories = open_hn.query_to_pandas_safe(query)
unique_stories
# query to SELECT and COUNT all the deleted comments
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted IS NOT NULL
        """

# perhaps a simpler way to count the deleted comments?
query3 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
deleted_comments = open_hn.query_to_pandas_safe(query2)
deleted_comments