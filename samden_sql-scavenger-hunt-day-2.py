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
# query to pass to 
myquery1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(ID) > 1000
        """
# <=1 GB by default
type_stories = hacker_news.query_to_pandas_safe(myquery1)
type_stories
myquery2 = """SELECT deleted, count(ID)
            FROM `bigquery-public-data.hacker_news.comments`
            Group by deleted
        """
# <=1 GB by default
delete_stories = hacker_news.query_to_pandas_safe(myquery2)
delete_stories
myquery3 = """SELECT deleted, countif(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            Group by deleted
        """
# <=1 GB by default
delete_stories_new = hacker_news.query_to_pandas_safe(myquery3)
delete_stories_new