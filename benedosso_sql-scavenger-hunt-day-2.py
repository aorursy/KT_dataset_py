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
import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
query2 = """SELECT COUNT(id)
FROM `bigquery-public-data.hacker_news.full`"""
count_stories = hacker_news.query_to_pandas_safe(query2)
count_stories
query3 = """SELECT COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = True
"""
deleted_stories = hacker_news.query_to_pandas_safe(query3)
deleted_stories
#Find the top ten authors by average "score"
query4 = """SELECT author, AVG(score)
FROM `bigquery-public-data.hacker_news.stories`
GROUP BY author
ORDER BY AVG(score) DESC
LIMIT 10
"""
avg_score = hacker_news.query_to_pandas_safe(query4)
avg_score