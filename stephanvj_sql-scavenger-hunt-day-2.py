# Your code goes here :)
import numpy as np
import pandas as pd
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query_stories = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """

story_count = hacker_news.query_to_pandas_safe(query_stories)
#Save answer for question 1
story_count.to_csv('stories.csv', index=True)

query_deleted = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY deleted
            """
deleted = hacker_news.query_to_pandas_safe(query_deleted)
#Save answer for question 2
deleted.to_csv('deleted.csv', index=True)
