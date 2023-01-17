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
query1 = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
"""

df1 = hacker_news.query_to_pandas_safe(query1)

df1
query2 = """SELECT deleted, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
"""

df2 = hacker_news.query_to_pandas_safe(query2)

df2
query3 = """SELECT type, COUNT(id) as count, AVG(descendants) as avg_descendants
            FROM `bigquery-public-data.hacker_news.full`
            WHERE descendants IS NOT NULL
            GROUP BY type
"""

df3 = hacker_news.query_to_pandas_safe(query3)

df3