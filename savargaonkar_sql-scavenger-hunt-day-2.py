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
# Query for stories of each type
query1= """SELECT type, Count(id) 
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
story_type= hacker_news.query_to_pandas_safe(query1)
story_type.head()
# Query for counting deleted comments
query2 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            """
deleted_comments = hacker_news.query_to_pandas_safe(query2)
deleted_comments.head()
# Query for counting deleted comments
query3 = """SELECT parent, COUNT(id) AS TotalComments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent HAVING MIN(ranking) > 0 """
groupcon_comments = hacker_news.query_to_pandas_safe(query3)
groupcon_comments.head()