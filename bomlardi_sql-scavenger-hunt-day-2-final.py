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
#count of type

type_query = """SELECT type, COUNT(type) as count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
                """
type_count = hacker_news.query_to_pandas_safe(type_query, .15)
type_count
#deleted comments
del_query = """ SELECT deleted, COUNT(id) as count
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY deleted"""
deleted_comms = hacker_news.query_to_pandas_safe(del_query, .1)
deleted_comms
#bonus stuff
#average score
avg_query = """ SELECT avg(score) as average, type
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type"""
average_score = hacker_news.query_to_pandas_safe(avg_query, .2)
average_score