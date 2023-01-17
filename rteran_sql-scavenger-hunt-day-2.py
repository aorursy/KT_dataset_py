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
# Your code goes here :)
query = """SELECT 
                type, 
                COUNT(id) AS count
            FROM 
                `bigquery-public-data.hacker_news.full`
            GROUP BY 
                type
            ORDER BY
                count DESC
        """
stories_by_type = hacker_news.query_to_pandas_safe(query)
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
stories_by_type.head()
query = """SELECT  
                COUNT(id) AS deleted_comments
            FROM 
                `bigquery-public-data.hacker_news.full`
            WHERE
                type = 'comment'
                AND deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
# How many comments have been deleted?
deleted_comments
