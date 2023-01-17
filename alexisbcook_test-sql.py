from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "stackoverflow" dataset

dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Your code here

bigquery_experts_query = """

                         SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers

                         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

                         INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                             ON q.id = a.parent_Id

                         WHERE q.tags LIKE '%bigquery%'

                         GROUP BY a.owner_user_id

                         """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

bigquery_experts_query_job = client.query(bigquery_experts_query, job_config=safe_config)

bigquery_experts_answer = bigquery_experts_query_job.to_dataframe()



bigquery_experts_answer.head()
# Your code here

bigquery_experts_query = """

                         SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers  

                         FROM `bigquery-public-data.stackoverflow.posts_questions` AS q    

                         INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a  

                             ON q.id = a.parent_Id

                         WHERE q.tags LIKE '%bigquery%'   

                         GROUP BY a.owner_user_id   

                         """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

bigquery_experts_query_job = client.query(bigquery_experts_query, job_config=safe_config)

bigquery_experts_answer = bigquery_experts_query_job.to_dataframe()



bigquery_experts_answer.head()