



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import our bq_helper package

import bq_helper

# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset

hacker_news.list_tables()
hacker_news.table_schema("full") 
# this query looks in the full table in the hacker_news

# dataset, then gets the score column from every row where 

# the type column has "job" in it.

query = """SELECT score

            FROM `bigquery-public-data.hacker_news.full`

            WHERE type = "job" """



# check how big this query will be

hacker_news.estimate_query_size(query)


# only run this query if it's less than 100 MB

hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the 

# query is smaller than 1 gig)

#ob_post_scores = hacker_news.query_to_pandas_safe(query)
#Since this has returned a dataframe, we can work with it as

#we would any other dataframe. For example, we can get the mean of the column:



# average score for job posts

#job_post_scores.score.mean()