# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# How many stories are there of each type in the full table?

query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

all_types = hacker_news.query_to_pandas_safe(query1)
all_types.head()
# How many comments have been deleted?

query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
         """
del_comm = hacker_news.query_to_pandas_safe(query2)
del_comm.head()
# Modify one of the previous queries to utilize a function other than COUNT().

query3 = """SELECT deleted, COUNTIF(id > 0)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
         """
del_comm2 = hacker_news.query_to_pandas_safe(query3)
del_comm2.head()