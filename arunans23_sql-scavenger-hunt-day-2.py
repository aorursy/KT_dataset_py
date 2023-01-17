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

#answer for question 1

query1 = """
            SELECT type, COUNT(id) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
"""

type_count = hacker_news.query_to_pandas_safe(query1)

type_count.head()

#answer for question 2

query2 = """
            SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
"""

delete_count = hacker_news.query_to_pandas_safe(query2)

delete_count.head()

#answer for question 3

query2 = """
            SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
"""

delete_count = hacker_news.query_to_pandas_safe(query2)

delete_count.head()

#Answer for question 3

query3 = """
            SELECT COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
"""

delete_count = hacker_news.query_to_pandas_safe(query2)

delete_count.head()

