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
hacker_news.table_schema(table_name="full")
hacker_news.head(table_name="full")
##How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """

stories_grouped_by_type_df = hacker_news.query_to_pandas_safe(query=query1)

stories_grouped_by_type_df.head()
# How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

query_2 = """SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING deleted = True
          """

count_of_deleted_comments = hacker_news.query_to_pandas_safe(query=query_2)

count_of_deleted_comments.head()
#Optional extra credit: read about aggregate functions other than COUNT() 
#and modify one of the queries you wrote above to use a different aggregate function.
query_3 = """SELECT deleted, COUNTIF(deleted)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING deleted = True
          """

count_of_deleted_comments = hacker_news.query_to_pandas_safe(query=query_3)

count_of_deleted_comments.head()
print("Estimated size of \"query_2\" which uses the aggregate function \"COUNT\" = {}GB"
      .format(hacker_news.estimate_query_size(query=query_2)))
print("Estimated size of \"query_3\" which uses the aggregate function \"COUNTIF\" = {}GB"
      .format(hacker_news.estimate_query_size(query=query_3)))