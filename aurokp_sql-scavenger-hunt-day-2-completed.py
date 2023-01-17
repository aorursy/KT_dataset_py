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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to 
query_no_stories = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Query the dataset
number_stories = hacker_news.query_to_pandas_safe(query_no_stories)

# Print 
number_stories
query_deleted = """SELECT deleted, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            where deleted = True
            GROUP BY deleted
        """

# Query the dataset
number_deleted = hacker_news.query_to_pandas_safe(query_deleted)

# Print
number_deleted
# USing countIf to count the number of deleted columns

query_deleted_countif = """SELECT COUNTIF(deleted = True) as deleted
            FROM `bigquery-public-data.hacker_news.full`
        """

# Query the dataset
number_deleted_countif = hacker_news.query_to_pandas_safe(query_deleted_countif)

# Print
number_deleted_countif