# import package with helper functions 
import bq_helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
hacker_news.head('full')


# query to pass to 
query2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
stories_by_type = hacker_news.query_to_pandas_safe(query2)
stories_by_type
stories_by_type.to_csv("stories_by_type.csv")
sns.factorplot('type', data= stories_by_type, hue='f0_',palette='coolwarm' , kind='count')
# query to pass to 
query3 = """SELECT deleted, COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True          
            
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted = hacker_news.query_to_pandas_safe(query3)
deleted.to_csv("deleted.csv")
deleted
# query to pass to 
# looking for only COMMENTS that were deleted
query4 = """SELECT type, COUNT(deleted = True)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = "comment"          
            
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
comments_deleted = hacker_news.query_to_pandas_safe(query4)
comments_deleted
comments_deleted.to_csv("comments_deleted.csv")