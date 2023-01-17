
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



import bq_helper
# helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",
                                      dataset_name="hacker_news")
#print the tables in the dataset
hacker_news.list_tables()
#print the 'full' table
hacker_news.table_schema("full")
#Print a couple of rows in the table
hacker_news.head("full")
hacker_news.head("full", selected_columns="by", num_rows=10)
#This query will get the score column where every row has the "job" in the column
query = """SELECT score
             FROM `bigquery-public-data.hacker_news.full`
             WHERE type= "job" """

#Check query size
hacker_news.estimate_query_size(query)
#will only run if less tha 100mb
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.score.mean()