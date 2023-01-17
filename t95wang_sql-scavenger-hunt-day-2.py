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
# Number of story tpyes
types = """SELECT type, COUNT(id) AS counts
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
          """
stories_types = hacker_news.query_to_pandas_safe(types)
stories_types.head()

# Number of comments have been deleted
deleted = """SELECT deleted, COUNT(id) AS counts
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
          """
comment_deleted = hacker_news.query_to_pandas_safe(deleted)
comment_deleted.head()
socre = """SELECT MIN(score) AS min, AVG(score) AS avg, MAX(score) AS max
           FROM `bigquery-public-data.hacker_news.full`
        """
story_socre = hacker_news.query_to_pandas_safe(socre)
story_socre