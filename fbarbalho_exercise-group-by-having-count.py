# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")
query = """SELECT type,COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_type = hacker_news.query_to_pandas_safe(query)
print(stories_type)
query = """SELECT COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True
        """

deleted_count = hacker_news.query_to_pandas_safe(query)
print(deleted_count)
query = """SELECT deleted, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
        """

deleted_group = hacker_news.query_to_pandas_safe(query)
print(deleted_group)
