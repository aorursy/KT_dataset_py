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
hacker_news.table_schema("full")
# import package with helper functions 
#import bq_helper

# create a helper object for this dataset
#hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")


# print the first couple rows of the "full" table
hacker_news.head("full")
# query to pass to 
query_types = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(id) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 300 MB by default
df_types = hacker_news.query_to_pandas_safe(query_types, 0.3)
df_types.head(10)
#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.table_schema("comments")
# query to pass to 
query_count_deleted_comment_1 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 300 MB
df_types = hacker_news.query_to_pandas_safe(query_count_deleted_comment_1, 0.3)

df_types
#Alternative way - by using GROUP and having "deleted" column in the result:
# query to pass to 
query_count_deleted_comment_2 = """SELECT deleted, COUNT(id) AS cnt
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 300 MB
df_types2 = hacker_news.query_to_pandas_safe(query_count_deleted_comment_2, 0.3)

df_types2
query_count_deleted_comment_3 = """SELECT deleted, COUNTIF(deleted) AS cnt
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """

df_types3 = hacker_news.query_to_pandas_safe(query_count_deleted_comment_3, 0.3)

df_types3
query_size1 = hacker_news.estimate_query_size(query = query_count_deleted_comment_1)
query_size2 = hacker_news.estimate_query_size(query = query_count_deleted_comment_2)
query_size3 = hacker_news.estimate_query_size(query_count_deleted_comment_3)
print("Estimated query's sizes. Simple COUNT: {}, GROUP BY/COUNT: {}, GROUP BY/COUNTIF: {}."
      .format(query_size1, query_size2, query_size3))