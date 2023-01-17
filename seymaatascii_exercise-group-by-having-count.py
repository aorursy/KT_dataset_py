# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex3 import *

print("Setup Complete")
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
# import package with helper functions 

import bq_helper



# create a helper object for this dataset

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="hacker_news")



# print the first couple rows of the "comments" table

hacker_news.head("full")
# Query to select prolific commenters and post counts

prolific_commenters_query = """

        SELECT author, COUNT(1) AS NumPosts

        From `bigquery-public-data.hacker_news.comments`

        GROUP BY author

        HAVING COUNT(1)>10000



"""

___ # Your code goes here



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(prolific_commenters_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

prolific_commenters = query_job.to_dataframe()



# View top few rows of results

print(prolific_commenters.head())



# Check your answer

q_1.check()
#q_1.solution()
# Write your query here and figure out the answer

query2 ="""

        SELECT COUNT(id)

        FROM `bigquery-public-data.hacker_news.comments`

        WHERE deleted is True

        """



deleted_count = hacker_news.query_to_pandas_safe(query2)

print(deleted_count.head())
#Displaying the max score for each type 

query3 ="""

        SELECT type, MAX(score) as MAX_Score

        FROM `bigquery-public-data.hacker_news.full`

        GROUP BY type

        """



max_score = hacker_news.query_to_pandas_safe(query3)

print(max_score.head())
# **Optional extra credit**: read about [aggregate functions other than COUNT()]

# and modify one of the queries you wrote above to use a different aggregate function.

query4 = """SELECT COUNTIF(deleted = True), deleted

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY deleted

        """

comments_deleted_new = hacker_news.query_to_pandas_safe(query4)

#q_2.solution()