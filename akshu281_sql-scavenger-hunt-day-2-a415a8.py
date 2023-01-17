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
# Your code goes here :)

# import package with helper functions 
import bq_helper
#from bq_helper import BigQueryHelper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query = """SELECT COUNT(id) AS Total_IDs,COUNTIF(deleted=True) AS Deleted_Count
            FROM `bigquery-public-data.hacker_news.comments`
            
        """
deleted_count = hacker_news.query_to_pandas_safe(query)
deleted_count.head()

#BigQueryHelper.table_schema("comments")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
types_count = hacker_news.query_to_pandas_safe(query1)
types_count.head()