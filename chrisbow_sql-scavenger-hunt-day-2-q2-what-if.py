# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# let's have a look at the first few rows of the comments table
hacker_news.head("comments")
# backticks around `by` as it is an SQL function

query6 = """SELECT `by`, author, id, deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted=True
         """
# submit query using the safe, scan-size limited function
deletedBy = hacker_news.query_to_pandas_safe(query6)

# print the result
deletedBy.head