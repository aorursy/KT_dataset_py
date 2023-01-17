# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first couple rows of the "comments" table
hacker_news.head("comments")


# Your Code Here
# Your Code Here
# Your Code Here