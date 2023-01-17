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

list_of_tables = [t.table_id for t in list(client.list_tables(dataset))]# Your code here



# Print your answer

print(list_of_tables)



# Check your answer

q_1.check()
#q_1.solution()
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
q_2.solution()
# Your code here

questions_query = """

                  SELECT id, title, owner_user_id

                  FROM `bigquery-public-data.stackoverflow.posts_questions`

                  WHERE tags like '%bigquery%'

                  """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

questions_query_job = client.query(questions_query, job_config = safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

questions_results = questions_query_job.to_dataframe() # Your code goes here



# Preview results

print(questions_results.head())



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
# Your code here

answers_query = """

                SELECT PA.id, PA.body, PA.owner_user_id

                FROM `bigquery-public-data.stackoverflow.posts_answers` as PA

                 INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` as PQ

                 ON PA.parent_id = PQ.id

                WHERE PQ.tags like '%bigquery%'



                """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=20**10)

answers_query_job = client.query(answers_query, job_config = safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

answers_results = answers_query_job.to_dataframe() # Your code goes here



# Preview results

print(answers_results.head())



# Check your answer

q_4.check()
#q_4.hint()

#q_4.solution()
# Your code here

bigquery_experts_query = """

                SELECT PA.owner_user_id AS user_id, count(1) AS number_of_answers

                FROM `bigquery-public-data.stackoverflow.posts_answers` as PA

                 INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` as PQ

                 ON PA.parent_id = PQ.id

                WHERE PQ.tags like '%bigquery%'

                GROUP BY PA.owner_user_id

                """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

bigquery_experts_query_job = client.query(bigquery_experts_query, job_config = safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

bigquery_experts_results = bigquery_experts_query_job.to_dataframe() # Your code goes here



# Preview results

print(bigquery_experts_results.head())



# Check your answer

q_5.check()
#q_5.hint()

#q_5.solution()
q_6.solution()