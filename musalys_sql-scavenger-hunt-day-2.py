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

# query for question 1

query_1 = """
                SELECT COUNT(id) as cnt_id, type
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                ORDER BY cnt_id
            """
hacker_news.estimate_query_size(query_1)
# extract result data through pandas dataframe
hacker_news_query1_df = hacker_news.query_to_pandas_safe(query_1)
hacker_news_query1_df.head()
# query for question 2
query_2 = """
            SELECT COUNT(id) as cnt_id, deleted
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            ORDER BY cnt_id
"""
hacker_news.estimate_query_size(query_2)
# extract result through dataframe
hacker_news_query2_df = hacker_news.query_to_pandas_safe(query_2)
hacker_news_query2_df.head()
# extra question
# how many stories 

query_3 = """
                SELECT id, SUM(parent) as sum_ranking
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY parent
"""

hacker_news.estimate_query_size(query_3)