# set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex6 import *

print("Setup is completed")
# create a "Client" object

from google.cloud import bigquery

client = bigquery.Client()



# construct a reference to the "stackoverflow" dataset

dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# get a list of available tables 

list_of_tables = [table.table_id for table in client.list_tables(dataset)]



# print your answer

print(list_of_tables)



# check your answer

q_1.check()
# for the solution, uncomment the line below.

# q_1.solution()
# construct a reference to the "posts_answers" table

answers_table_ref = dataset_ref.table("posts_answers")



# API request - fetch the table

answers_table = client.get_table(answers_table_ref)



# preview the first five lines of the "posts_answers" table

client.list_rows(answers_table, max_results=5).to_dataframe()
# construct a reference to the "posts_questions" table

questions_table_ref = dataset_ref.table("posts_questions")



# API request - fetch the table

questions_table = client.get_table(questions_table_ref)



# preview the first five lines of the "posts_questions" table

client.list_rows(questions_table, max_results=5).to_dataframe()
q_2.solution()
# your code here

questions_query = """

                  SELECT id, title, owner_user_id

                  FROM `bigquery-public-data.stackoverflow.posts_questions`

                  WHERE tags LIKE '%bigquery%'

                  """



# set up the query (cancel the query if it would use too much of your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

questions_query_job = client.query(questions_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

questions_results = questions_query_job.to_dataframe()



# preview results

print(questions_results.head())



# check your answer

q_3.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_3.hint()

# q_3.solution()
# your code here

answers_query = """

                SELECT a.id, a.body, a.owner_user_id

                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q 

                INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                    ON q.id = a.parent_id

                WHERE q.tags LIKE '%bigquery%'

                """



# set up the query (cancel the query if it would use too much of your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

answers_query_job = client.query(answers_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

answers_results = answers_query_job.to_dataframe()



# preview results

print(answers_results.head())



# check your answer

q_4.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_4.hint()

# q_4.solution()
# your code here

bigquery_experts_query = """

                         SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers

                         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                         INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                             ON q.id = a.parent_id

                         WHERE q.tags LIKE '%bigquery%'

                         GROUP BY user_id

                         """



# set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

bigquery_experts_query_job = client.query(bigquery_experts_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

bigquery_experts_results = bigquery_experts_query_job.to_dataframe()



# preview results

print(bigquery_experts_results.head())



# Check your answer

q_5.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_5.hint()

# q_5.solution()
q_6.solution()