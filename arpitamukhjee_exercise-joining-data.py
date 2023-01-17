# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex6 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "stackoverflow" dataset

dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Get a list of available tables 

tables = list(client.list_tables(dataset))

list_of_tables = [table.table_id for table in tables] 



# Print your answer

print(list_of_tables)



# Check your answer

q_1.check()
# Construct a reference to the "posts_answers" table

answers_table_ref = dataset_ref.table("posts_answers")



# API request - fetch the table

answers_table = client.get_table(answers_table_ref)



# Preview the first five lines of the "posts_answers" table

client.list_rows(answers_table, max_results=5).to_dataframe()
# Construct a reference to the "posts_questions" table

questions_table_ref = dataset_ref.table("posts_questions")



# API request - fetch the table

questions_table = client.get_table(questions_table_ref)



# Preview the first five lines of the "posts_questions" table

client.list_rows(questions_table, max_results=5).to_dataframe()
query = """

SELECT PA.owner_user_id AS id, COUNT(PQ.tags) AS topic

FROM `bigquery-public-data.stackoverflow.posts_questions` AS PQ

INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS PA

ON PQ.owner_user_id=PA.owner_user_id

GROUP BY PA.owner_user_id



"""

# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

query_results = query_job.to_dataframe()



# Preview results

print(query_results.head())



# Check your answer

q_3.check()
q_2.solution()
answers_query = """

SELECT PA.id, PA.body, PA.owner_user_id

                  FROM `bigquery-public-data.stackoverflow.posts_questions` AS PQ

                  INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS PA

                  ON PQ.id=PA.parent_id

                  WHERE PQ.tags LIKE '%bigquery%'

                 """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)

answers_query_job = client.query(answers_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

answers_results = answers_query_job.to_dataframe()



# Preview results

print(answers_results.head())



# Check your answer

q_4.check()
answers_query = """

SELECT PA.id, PA.body, PA.owner_user_id

                  FROM `bigquery-public-data.stackoverflow.posts_questions` AS PQ

                  INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS PA

                  ON PQ.id=PA.parent_id

                  WHERE PQ.tags LIKE '%bigquery%'

                 """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

answers_query_job = client.query(answers_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

answers_results = answers_query_job.to_dataframe()



# Preview results

print(answers_results.head())



# Check your answer

q_4.check()
# Your code here

bigquery_experts_query = """

                SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q 

                INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                    ON q.id = a.parent_id

                WHERE q.tags LIKE '%bigquery%'

                GROUP BY a.owner_user_id

                """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

bigquery_experts_query_job = client.query(bigquery_experts_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame

bigquery_experts_results = bigquery_experts_query_job.to_dataframe()



# Preview results

print(bigquery_experts_results.head())



# Check your answer

q_5.check()