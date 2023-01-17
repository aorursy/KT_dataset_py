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
query_q1 = """ SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
            """
hacker_news.query_to_pandas_safe(query_q1)
query_q2 = """ SELECT deleted, COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY deleted
                HAVING deleted = True
            """
hacker_news.query_to_pandas_safe(query_q2)
# find the time when the bigger contributor (that wrote more than 10 thousand comments), wrote the last comment.
query_q3 = """ SELECT author, MAX(time)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY author
                HAVING COUNT(id) > 10000
                ORDER BY MAX(time) DESC
            """

hacker_news.query_to_pandas_safe(query_q3)