# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper


hacker=bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='hacker_news')
hacker.list_tables()
hacker.table_schema('full')
hacker.head('full')
query=""" SELECT count(*) as count_story,type from `bigquery-public-data.hacker_news.full` GROUP BY type HAVING type= 'story' """
hacker.estimate_query_size(query)
story=hacker.query_to_pandas_safe(query)
story
hacker.table_schema('comments')
hacker.head('comments')
query=""" SELECT count(*) as count_deleted,deleted from `bigquery-public-data.hacker_news.comments` group by deleted having deleted is True """ 
hacker.estimate_query_size(query)
deleted=hacker.query_to_pandas_safe(query)
deleted
hacker.table_schema('stories')
hacker.head('stories')
query="""SELECT count(id) as id_count,score from `bigquery-public-data.hacker_news.stories` group by score having score = 1 """
hacker.estimate_query_size(query)
stories=hacker.query_to_pandas_safe(query)
stories
query = """ SELECT sum(score) as sum_score,deleted from `bigquery-public-data.hacker_news.stories` group by deleted """
hacker.estimate_query_size(query)
sum_story=hacker.query_to_pandas_safe(query)
sum_story