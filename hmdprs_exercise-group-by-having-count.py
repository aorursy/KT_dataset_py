# set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex3 import *

print("Setup is completed")
# create a "Client" object

from google.cloud import bigquery

client = bigquery.Client()



# construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# construct a reference to the "comments" table

table_ref = dataset_ref.table("comments")



# API request - fetch the table

table = client.get_table(table_ref)



# preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
# query to select prolific commenters and post counts

prolific_commenters_query = """

                            SELECT author, COUNT(1) AS NumPosts

                            FROM `bigquery-public-data.hacker_news.comments`

                            GROUP BY author

                            HAVING COUNT(1) > 10000

                            """



# set up the query (cancel the query if it would use too much of your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(prolific_commenters_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

prolific_commenters = query_job.to_dataframe()



# preview top few rows of results

print(prolific_commenters.head())



# check your answer

q_1.check()
# for the solution, uncomment the line below

# q_1.solution()
# write your query here and figure out the answer

deleted_comments_query = """

                         SELECT COUNT(1) AS NumDeletedComments

                         FROM `bigquery-public-data.hacker_news.comments`

                         WHERE deleted = True

                         """



# set up the query (cancel the query if it would use too much of your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(deleted_comments_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

deleted_comments = query_job.to_dataframe()
# print the result

deleted_comments
num_deleted_posts = 227736



q_2.check()
# for the solution, uncomment the line below.

# q_2.solution()