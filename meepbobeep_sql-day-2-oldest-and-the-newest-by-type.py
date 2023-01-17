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
hacker_news.head("full")
# Question 1:How many stories (use the "id" column) are there are each type in the full table?

query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_by_type = hacker_news.query_to_pandas_safe(query1)
count_by_type



# Question 2: How many comments have been deleted?
query2 = """ SELECT COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING deleted = True
            """
deleted_coms = hacker_news.query_to_pandas_safe(query2)
deleted_coms
# LAST: read about [aggregate functions other than COUNT()]
# and modify one of the queries you wrote above to use a different aggregate function.

# For each entry type, I find the earliest and the latest item of that type
query3 = """SELECT type, MIN(timestamp), MAX(timestamp)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
earliest_latest = hacker_news.query_to_pandas_safe(query3)
earliest_latest