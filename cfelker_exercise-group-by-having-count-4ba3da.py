# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
full = hacker_news.query_to_pandas_safe(query)
print(full.head())
# query to pass to 
query = """SELECT parent COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE COUNT(id) > 10 
            HAVING deleted = True
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
full = hacker_news.query_to_pandas_safe(query)
print(full.head())
query = """SELECT MAX(parent)
        FROM `bigquery-public-data.hacker_news.full`
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
full = hacker_news.query_to_pandas_safe(query)
print(full.head())