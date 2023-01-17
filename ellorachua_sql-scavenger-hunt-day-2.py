# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass 
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
count_type = hacker_news.query_to_pandas_safe(query1)

# print the data frame to show the count of each type
count_type.head()
# print the first couple rows of the "full" table
hacker_news.head("comments")
# query to pass 
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
count_deleted = hacker_news.query_to_pandas_safe(query2)

# print the data frame to show the count of each type
count_deleted

# query to pass 
query3 = """SELECT deleted, COUNTIF(deleted)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
count_deleted_modified = hacker_news.query_to_pandas_safe(query3)

# print the data frame to show the count of each type
count_deleted_modified