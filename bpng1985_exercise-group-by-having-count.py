# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full_201510`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories)
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
comments = hacker_news.query_to_pandas_safe(query)
print(comments)
# Your Code Hereimport bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

query = """SELECT max(time_ts)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
comments = hacker_news.query_to_pandas_safe(query)
print(comments)