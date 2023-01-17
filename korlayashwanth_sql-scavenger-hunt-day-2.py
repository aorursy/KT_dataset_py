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
#importing the big query helper library
import bq_helper
#creating an object out of helper linrary
bq_object = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',dataset_name='hacker_news')
#To get the description about the data in the table
bq_object.table_schema('full')
#To get the first few rows of the table.
bq_object.head('full')
query1 = """
        SELECT type,count(id) from `bigquery-public-data.hacker_news.full` GROUP BY type
        """
bq_object.query_to_pandas_safe(query1)
query2 = """
        SELECT deleted, count(id) from `bigquery-public-data.hacker_news.comments` GROUP BY deleted HAVING deleted is TRUE
        """
bq_object.query_to_pandas_safe(query2)
#Optional

query3 = """
        SELECT type,count(id),MAX(parent) from `bigquery-public-data.hacker_news.full` GROUP BY type
        """
bq_object.query_to_pandas_safe(query3)