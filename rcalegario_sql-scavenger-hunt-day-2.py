# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print all the tables in this dataset
hacker_news.list_tables()
# print the first couple rows of the "stories" dataset
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_by_type = hacker_news.query_to_pandas_safe(query1)

stories_by_type
# print the first couple rows of the "stories" dataset
hacker_news.head("comments")
# query to pass to 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted IS NOT NULL
        """

deleted_comments = hacker_news.query_to_pandas_safe(query2)
# number of comments that have been deleted
deleted_comments
query_op2 = """SELECT COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
        """

deleted_comments_op = hacker_news.query_to_pandas_safe(query_op2)

deleted_comments_op