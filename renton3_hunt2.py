import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                      dataset_name = 'hacker_news')
hacker_news.table_schema('full')
hacker_news.head("full")
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""
stories_of_each_type = hacker_news.query_to_pandas_safe(query1)
stories_of_each_type.head()
query2 = """SELECT COUNT(deleted)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted IS True
            """
deleted_coments1 = hacker_news.query_to_pandas_safe(query2)
deleted_coments1.head()
query3 = """SELECT deleted, count(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted IS True"""
deleted_coments2 = hacker_news.query_to_pandas_safe(query3)
deleted_coments2.head()