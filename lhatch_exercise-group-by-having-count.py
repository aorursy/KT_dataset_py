# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_type_count = hacker_news.query_to_pandas_safe(query)
story_type_count
# query to pass to 
query = """SELECT type, deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type, deleted
            HAVING type = 'comment' and deleted
        """

deleted_comments_count = hacker_news.query_to_pandas_safe(query)
deleted_comments_count
# query to pass to 
query = """SELECT type, deleted, COUNTIF(deleted) as count_deleted, COUNT(deleted is null) as count_notdeleted
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type, deleted
        """

any_deleted_count = hacker_news.query_to_pandas_safe(query)
any_deleted_count