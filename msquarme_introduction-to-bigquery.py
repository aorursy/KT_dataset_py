# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


### --------- Commented by me!!

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#print("Hello Big Query!")
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                      dataset_name = "hacker_news")

hacker_news.head('full', selected_columns="by", num_rows=11)

query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """
hacker_news.estimate_query_size(query)
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.score.mean()
job_post_scores.to_csv("Job_post_scores")
