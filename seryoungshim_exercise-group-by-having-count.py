# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
story = hacker_news.query_to_pandas_safe(query)
print(story.head())
# Your Code Here
query = """SELECT deleted=True, COUNT(id)
           From `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           HAVING deleted=True"""
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())
# Your Code Here
query = """SELECT deleted=True AS deleted, COUNTIF(deleted=True) AS count
           From `bigquery-public-data.hacker_news.comments`
           GROUP BY deleted
           HAVING COUNTIF(deleted=True)>0
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())