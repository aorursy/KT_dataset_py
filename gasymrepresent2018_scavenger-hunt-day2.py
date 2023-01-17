
import pandas as pd 

# import package with helper functions
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
dataset_name="hacker_news")

# print all the tables in this dataset 
hacker_news.list_tables()


# print the first couple rows of the "comments" table
hacker_news.head("comments")

# print the first couple rows of the "full" table
hacker_news.head("full_201510")
# print the first couple rows of the "full" table
hacker_news.head("stories")


numbStories_query = """SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set
# to 1 GB by default
numbStories = hacker_news.query_to_pandas_safe(numbStories_query)

#Output the number of stories of each type
numbStories
commentsDel_query = """SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
HAVING deleted = True
"""
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set
# to 1 GB by default
numbCommDel = hacker_news.query_to_pandas_safe(commentsDel_query)

#Output the number of stories of each type
numbCommDel
