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
stories_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_type = hacker_news.query_to_pandas_safe(stories_query)

stories_type.columns = ['type','Count ID']
stories_type


deleted_query = """SELECT deleted, count(deleted)
FROM `bigquery-public-data.hacker_news.full`
WHERE type = 'comment'
GROUP BY deleted
"""


deleted_comment = hacker_news.query_to_pandas_safe(deleted_query)
deleted_comment.columns = ['deleted','Count']
deleted_comment.head()