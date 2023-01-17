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
hacker_news.list_tables()
hacker_news.table_schema('comments')

query_1 = """SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type"""

hacker_news.estimate_query_size(query_1)    # 16373180   8399417   1959809
count_by_type = hacker_news.query_to_pandas_safe(query_1)
count_by_type

query_2 = """SELECT COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted = TRUE"""

hacker_news.estimate_query_size(query_2)
hacker_news.query_to_pandas_safe(query_2)   #returns 227736

query_3 = """SELECT AVG(CASE WHEN deleted THEN 1 ELSE 0 END)
                FROM `bigquery-public-data.hacker_news.comments`
                """
hacker_news.estimate_query_size(query_3)
hacker_news.query_to_pandas_safe(query_3) #returned 0.027113

