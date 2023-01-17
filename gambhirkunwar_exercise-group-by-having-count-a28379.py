# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
#hacker_news.list_tables()
hacker_news.head("full")
# stories type in the full table 
query = """ SELECT DISTINCT type, Count(id) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_id = hacker_news.query_to_pandas_safe(query)
print(type_id.head())
# no. of comments deleted
query = """ SELECT count(id) AS no_of_comments_deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
       """
no_deleted = hacker_news.query_to_pandas_safe(query)
print(no_deleted.head())
# Your Code Here
query = """ SELECT max(parent) AS max_parent
            FROM `bigquery-public-data.hacker_news.full`
       """
max_parent = hacker_news.query_to_pandas_safe(query)
print(no_deleted.head())