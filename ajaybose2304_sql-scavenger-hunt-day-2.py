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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# List of different types and number of occurences in full table

query1 = """SELECT  type as TYPE, count(id) as COUNT
            FROM  `bigquery-public-data.hacker_news.full`
            GROUP BY type"""

hacker_news.estimate_query_size(query1)
query1_ans = hacker_news.query_to_pandas_safe(query1, max_gb_scanned=0.3)



# Deleted count

query2 = """SELECT  count(id) as DELETED_COUNT
            FROM  `bigquery-public-data.hacker_news.comments`
            where deleted is True
        """

hacker_news.estimate_query_size(query2)
query2_ans = hacker_news.query_to_pandas_safe(query2, max_gb_scanned=0.4)

print(query1_ans)
print(query2_ans)