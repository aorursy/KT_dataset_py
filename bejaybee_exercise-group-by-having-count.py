# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Your Code Here
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY type
   
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
print(popular_stories.head())
# Your Code Here
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
print(popular_stories.head())
# Your Code Here
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """