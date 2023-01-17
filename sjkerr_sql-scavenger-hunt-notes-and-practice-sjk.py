# Just in case
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Notes From Handbook
# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")

# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")

# preview the first couple lines of the "full" table
hacker_news.head("full")

# preview the first ten entries in the by column of the full table
hacker_news.head("full", selected_columns="by", num_rows=10)

# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be - this is in GB
hacker_news.estimate_query_size(query)

# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)

# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)

# average score for job posts
job_post_scores.score.mean()

# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")

######################################################################################



