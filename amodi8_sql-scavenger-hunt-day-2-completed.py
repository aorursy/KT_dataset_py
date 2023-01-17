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
# Query to count number of stories of each type in the full table
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Results from the query stored in type_count dataframe
type_count = hacker_news.query_to_pandas_safe(query1)
# Preview of contents of type_count dataframe
type_count.head()
# Query to find out the number of deleted comments in the full table
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE 
        """
# Results from the query stored in deleted_comments dataframe
deleted_comments = hacker_news.query_to_pandas_safe(query2)
# Preview of contents of deleted_comments dataframe
deleted_comments.head()