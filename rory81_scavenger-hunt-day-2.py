# import package with helper functions
import bq_helper

#create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name = "hacker_news")

#print the first couple of rows of the "comments" table
hacker_news.head("comments")

# Number of comments that were made as responses to a specific comment 
query = """SELECT parent, COUNT(id) as number
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()

# How many stories are there of each type
stories = """SELECT type, COUNT(id) as number
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
          """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_stories = hacker_news.query_to_pandas_safe(stories)
print(type_stories)

# Number of comments deleted
comments = """SELECT count(id) as number_deleted
              FROM `bigquery-public-data.hacker_news.comments`
              WHERE deleted= True
           """

deleted_comments = hacker_news.query_to_pandas_safe(comments)
print(deleted_comments)
# What is the maximum and minimum score of a comment 
stories = """SELECT max(score) as max, min(score) as min
           FROM `bigquery-public-data.hacker_news.full`
          """

score_stories = hacker_news.query_to_pandas_safe(stories)
print(score_stories)