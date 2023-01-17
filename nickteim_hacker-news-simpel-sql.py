# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import our bq_helper package

import bq_helper 
from google.cloud import bigquery
# create a helper object for our bigquery dataset

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "comments" table

table_ref = dataset_ref.table("comments")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
# print a list of all the tables in the hacker_news dataset

hacker_news.list_tables()
# print information on all the columns in the "full" table

# in the hacker_news dataset

hacker_news.table_schema("full")
hacker_news.head("full")
# preview the first ten entries in the by column of the full table

hacker_news.head("full", selected_columns="by", num_rows=10)
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

job_post_scores = hacker_news.query_to_pandas_safe(query)
# Query to select countries with units of "ppm"

first_query = """

              SELECT parent

              FROM `bigquery-public-data.hacker_news.full`

              WHERE type = 'comment'

            """







# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

first_query_job = client.query(first_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

first_results = first_query_job.to_dataframe()

first_results.head()
query_improved = """

                 SELECT ranking, COUNT(1) AS NumRank 

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY ranking

                 HAVING COUNT(1) > 10

                 """

## AS means the name the data column vill have, parent er the column it will coint

## (COUNT(1)) We can use GROUP BY to group together rows that have the same value in 

##the Animal column, while using COUNT() to find out how many ID's we have in each group.

## HAVING is used in combination with GROUP BY to ignore groups that don't meet certain criteria.

## Here the COUNT cant be bigger 10.



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()
query_improved = """

                SELECT id, ranking, deleted

                FROM `bigquery-public-data.hacker_news.comments`

                ORDER BY id

                 """

##ORDER BY is usually the last clause in your query, and it sorts the results returned by the rest of your query.





safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()
query_improved = """

                 WITH time AS 

                 (

                     SELECT dead AS dead

                     FROM `bigquery-public-data.hacker_news.comments`

                 )

                 SELECT COUNT(1) AS transactions,

                        dead

                 FROM time

                 GROUP BY dead

                 ORDER BY dead

                 """



##A common table expression (or CTE) is a temporary table that you return

##within your query. CTEs are helpful for splitting your queries into readable 

##chunks, and you can write queries against them.



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

transactions_by_date = query_job.to_dataframe()



# Print the first five rows

transactions_by_date.head()
#Note the repos and dataset used in this code is not downloaded to this kernel, so it will note work

## However it vill show the concept if ever needed.



query = """

        SELECT L.license, COUNT(1) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 

            ON sf.repo_name = L.repo_name

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """

##In the query, ON determines which column in each table to use to combine the 

##tables. Notice that since the ID column exists in both tables, we have to clarify

##which one to use. We use p.ID to refer to the ID column from the pets table, and 

##o.Pet_ID refers to the Pet_ID column from the owners table.



##In general, when you're joining tables, it's a good habit to specify which table

##each of your columns comes from. That way, you don't have to pull up the schema 

##every time you go back to read the query.



##The type of JOIN we're using today is called an INNER JOIN. That means that a 

##row will only be put in the final output table if the value in the columns you're 

##using to combine them shows up in both the tables you're joining. For example, if

##Tom's ID number of 4 didn't exist in the pets table, we would only get 3 rows 

##back from this query. There are other types of JOIN, but an INNER JOIN is very

##widely used, so it's a good one to start with.





# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

file_count_by_license = query_job.to_dataframe()
# save our dataframe as a .csv 

job_post_scores.to_csv("job_post_scores.csv")