# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")
# Your Code Here
query = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
storycount = hacker_news.query_to_pandas_safe(query)
storycount.head()
# Your Code Here
query2 = """SELECT deleted, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY deleted
           HAVING True
        """
deletedcount = hacker_news.query_to_pandas_safe(query2)
print(deletedcount.head())
# Your Code Here
query3 = """SELECT COUNTIF(deleted = True) AS deleted_com
           FROM `bigquery-public-data.hacker_news.full`
        """
shortcount = hacker_news.query_to_pandas_safe(query3)
print(shortcount.head())