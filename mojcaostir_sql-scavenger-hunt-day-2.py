# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

query = """SELECT parent, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(id) > 10
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.list_tables()
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
hacker_news.estimate_query_size(query1)
stories = hacker_news.query_to_pandas_safe(query1)
stories
hacker_news.head("comments")
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
                GROUP BY deleted
         """
hacker_news.estimate_query_size(query2)
deleted = hacker_news.query_to_pandas_safe(query2)
deleted
query3 = """SELECT COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
         """
hacker_news.estimate_query_size(query3)
deleted
