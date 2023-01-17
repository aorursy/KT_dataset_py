# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
query = """ SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
group_type = hacker_news.query_to_pandas_safe(query)
group_type
hacker_news.head('comments')
query2 = """ SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
         """
deleted = hacker_news.query_to_pandas_safe(query2)
deleted
deleted_count = deleted.iloc[0,1]
deleted_count
query3 = """ SELECT deleted, MIN(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
         """
min_id = hacker_news.query_to_pandas_safe(query3)
min_id
query4 = """ SELECT deleted, COUNT(parent)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY deleted
             HAVING COUNT(parent)>1000000
         """  
deleted_1m = hacker_news.query_to_pandas_safe(query4)
deleted_1m # we do not have deleted comments that have parent > 1000000