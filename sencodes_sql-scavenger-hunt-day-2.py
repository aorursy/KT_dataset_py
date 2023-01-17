# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full", 10)
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
query_1 = """SELECT type, COUNT(id) as type_num
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
num_stories = hacker_news.query_to_pandas_safe(query_1)
num_stories.head()
query_2 = """SELECT COUNT(*) AS deleted_num
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = TRUE
            """
deleted_num = hacker_news.query_to_pandas_safe(query_2)
deleted_num
# modi_query_3 = """SELECT type, COUNT(id) as type_num
#                 FROM `bigquery-public-data.hacker_news.full`
#                 GROUP BY type
#                 """
# num_stories = hacker_news.query_to_pandas_safe(modi_query_3)
# num_stories.head()