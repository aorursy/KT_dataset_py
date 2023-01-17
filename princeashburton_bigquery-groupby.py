
import bq_helper

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))


hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")

hacker_news.head("comments")
query = """SELECT parent, COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           GROUP BY parent
           HAVING COUNT(id) > 10

        """
#We are setting a query limit here]
popular_stories = hacker_news.query_to_pandas_safe(query)
print(popular_stories.head())
query_type = """SELECT type, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
             
             
             """
story_type = hacker_news.query_to_pandas_safe(query_type)
print(story_type.head())
query_del = """SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY deleted
             HAVING deleted = True
            
            """
comments = hacker_news.query_to_pandas_safe(query_del)
print(comments)
query_rank = """SELECT  MAX(ranking) AS `Highest_Rank`, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY ranking

             """
ranking = hacker_news.query_to_pandas_safe(query_rank)
print(ranking)