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
hacker_news.head("full")

query1 = """SELECT type, COUNT(id) as Total
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY type
        """
type_cnt = hacker_news.query_to_pandas_safe(query1)
type_cnt
query2 = """SELECT deleted, COUNT(id) as Total
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
del_cnt = hacker_news.query_to_pandas_safe(query2)
del_cnt
query3 = """SELECT COUNT(*) as Total
            FROM `bigquery-public-data.hacker_news.full`
           WHERE deleted = True
        """
del_cnt = hacker_news.query_to_pandas_safe(query3)
del_cnt
# Retrieves Highest Score
query4 = """SELECT MAX(score) as MAX_SCORE
            FROM `bigquery-public-data.hacker_news.full`
        """
max_scr = hacker_news.query_to_pandas_safe(query4)
max_scr