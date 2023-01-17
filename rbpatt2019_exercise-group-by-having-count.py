# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
#there is no type column in this data, so I'm using the author column
query = """SELECT author, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING COUNT(id) > 10"""
data = hacker_news.query_to_pandas_safe(query)
data.describe()
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted"""
data = hacker_news.query_to_pandas_safe(query)
data
#finds oldest comment
query = """SELECT MIN(time)  
            FROM `bigquery-public-data.hacker_news.comments`"""
data = hacker_news.query_to_pandas_safe(query)
data