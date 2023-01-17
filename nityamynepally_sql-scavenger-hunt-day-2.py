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
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.head("comments")
query1 = """SELECT deleted ,COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            group by deleted
            Having count(deleted) > 100000  
        """
deleted_values = hacker_news.query_to_pandas_safe(query1)
deleted_values.head()

hacker_news.head("full")
query2 = """SELECT count(id),type
            FROM `bigquery-public-data.hacker_news.full`
            group by type 
        """
deleted_vs = hacker_news.query_to_pandas_safe(query2)
deleted_vs.head()
print(deleted_vs)

