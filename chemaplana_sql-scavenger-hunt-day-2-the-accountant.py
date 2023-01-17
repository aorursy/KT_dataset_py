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
l_tables = hacker_news.list_tables()
print (l_tables)
query_type = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.estimate_query_size(query_type)
story_types = hacker_news.query_to_pandas_safe(query_type)
print (story_types)
hacker_news.table_schema(l_tables[1])
query_deleted = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """
hacker_news.estimate_query_size(query_deleted)
deleted_stories = hacker_news.query_to_pandas_safe(query_deleted)
print (deleted_stories)
query_deleted2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
hacker_news.estimate_query_size(query_deleted2)
deleted_stories2 = hacker_news.query_to_pandas_safe(query_deleted2)
print (deleted_stories2)
query_deleted3 = """SELECT deleted
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
        """
hacker_news.estimate_query_size(query_deleted3)
deleted_stories3 = hacker_news.query_to_pandas_safe(query_deleted3)
print (len(deleted_stories3))
query_deleted4 = """SELECT COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.full`
        """
hacker_news.estimate_query_size(query_deleted4)
deleted_stories4 = hacker_news.query_to_pandas_safe(query_deleted4)
print (deleted_stories4)