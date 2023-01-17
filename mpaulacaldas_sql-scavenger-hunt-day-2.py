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
# print the first couple rows of the "comments" table
hacker_news.head("full")
# Query: count IDs by type
query_q1 = """SELECT type, COUNT(id) AS number_of_stories
              FROM `bigquery-public-data.hacker_news.full`
              GROUP BY type
           """
# Fetch results
stories_per_type = hacker_news.query_to_pandas_safe(query_q1)

# Print resulting data frame
stories_per_type
# Query: count number of comments deleted
query_q2 = """SELECT COUNT(id) AS comments_deleted
              FROM `bigquery-public-data.hacker_news.full`
              WHERE deleted = True AND type = 'comment'
           """
# Fetch results
comments_deleted = hacker_news.query_to_pandas_safe(query_q2)

# Print resulting data frame
comments_deleted
# Query
query_q2_v2 = """SELECT COUNTIF(deleted = True) AS number_deleted, type
                 FROM `bigquery-public-data.hacker_news.full`
                 GROUP BY type
              """
# Fetch results
comments_deleted_v2 = hacker_news.query_to_pandas_safe(query_q2_v2)

# Print resulting data frame
comments_deleted_v2