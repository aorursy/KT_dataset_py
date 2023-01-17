# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
print ('checking everthing is imported')
# Any results you write to the current directory are saved as output.
hacker_news = bq_helper.BigQueryHelper (active_project = 'bigquery-public-data',
                                       dataset_name ='hacker_news')
hacker_news.list_tables()
hacker_news.table_schema('full')
hacker_news.head('full', selected_columns=['by','id'], num_rows=10)
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """
hacker_news.estimate_query_size(query)
jobs_score = hacker_news.query_to_pandas_safe(query,max_gb_scanned=0.5)
jobs_score.score.mean()
jobs_score.to_csv('jobs_post_scores.csv')
