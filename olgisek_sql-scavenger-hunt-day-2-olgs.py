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
types_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
types = hacker_news.query_to_pandas_safe(types_query)
types.head()
# top types :)
top_types_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING COUNT(id) > 11000
        """
top_types = hacker_news.query_to_pandas_safe(top_types_query)
top_types.head()
# deleted comments
del_comm_query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
del_comm = hacker_news.query_to_pandas_safe(del_comm_query)
del_comm.head()
# average
average_ranking_query = """SELECT AVG(ranking)
            FROM `bigquery-public-data.hacker_news.full`
            """
average_ranking = hacker_news.query_to_pandas_safe(average_ranking_query)
average_ranking.head()