# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "full" table
hacker_news.head("full")
# set up query
query1 = """SELECT type, COUNT(id)
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """

print("Size of query:", hacker_news.estimate_query_size(query1))
types = hacker_news.query_to_pandas_safe(query1)
# print out the types
types
# preview head of the comments table
hacker_news.head("comments")
# construct query
query2 = """SELECT deleted, COUNT(ID)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
         """

print("Size of query:", hacker_news.estimate_query_size(query2))
deleted_comments = hacker_news.query_to_pandas_safe(query2)
# print
deleted_comments
# construct query
query3 = """SELECT deleted, COUNT(ID)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
            GROUP BY deleted
            HAVING deleted = True
         """

print("Size of query:", hacker_news.estimate_query_size(query3))
deleted_comments_full = hacker_news.query_to_pandas_safe(query3)
# print
deleted_comments_full
# construct query
query4 = """SELECT COUNTIF(deleted = True) as count_del
            FROM `bigquery-public-data.hacker_news.comments`
         """

print("Size of query:", hacker_news.estimate_query_size(query4))
deleted_comments_extra = hacker_news.query_to_pandas_safe(query4)
# print
deleted_comments_extra
# construct query
query5 = """SELECT COUNTIF(deleted = True) as count_del
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'comment'
         """

print("Size of query:", hacker_news.estimate_query_size(query5))
deleted_comments_extra_full = hacker_news.query_to_pandas_safe(query5)
# print
deleted_comments_extra_full