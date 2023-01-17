# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
Types = hacker_news.query_to_pandas_safe(query)
Types
# query to pass to 
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
Number_of_Deleted_Comments = hacker_news.query_to_pandas_safe(query)
Number_of_Deleted_Comments
# query to pass to 
query = """SELECT deleted, COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """
Number_of_Deleted_Comments = hacker_news.query_to_pandas_safe(query)
Number_of_Deleted_Comments