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
query_stories = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
popular_storiess = hacker_news.query_to_pandas_safe(query_stories)
print(popular_storiess)

query_deleted = """SELECT  deleted,COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted HAVING deleted = True
        """
comment_deleted = hacker_news.query_to_pandas_safe(query_deleted)
print(comment_deleted)

query_sum = """SELECT type, SUM(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
popular_sum = hacker_news.query_to_pandas_safe(query_sum)
print(popular_sum)

