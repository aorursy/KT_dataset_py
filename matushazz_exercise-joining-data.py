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

tables = list(client.list_tables(dataset)) # Your code here

list_of_tables = [table.table_id for table in tables]

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

                  SELECT id,title, owner_user_id

                  FROM `bigquery-public-data.stackoverflow.posts_questions`

                  WHERE tags like '%bigquery%'

                  """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

questions_query_job = client.query(questions_query, job_config =safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

questions_results = questions_query_job.to_dataframe() # Your code goes here



# Preview results

print(questions_results.head())



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
# Your code here

answers_query = """SELECT t.id, t.body, t.owner_user_id

                    FROM `bigquery-public-data.stackoverflow.posts_answers` t

                    INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` p

                    on t.parent_id = p.id 

                    WHERE p.tags like '%bigquery%'"""



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)

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

bigquery_experts_query = """SELECT t.owner_user_id as user_id, count(t.id) as number_of_answers

                    FROM `bigquery-public-data.stackoverflow.posts_answers` t

                    INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` p

                    on t.parent_id = p.id 

                    WHERE p.tags like '%bigquery%'

                    group by t.owner_user_id"""



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

bigquery_experts_query_job = client.query(bigquery_experts_query, safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

bigquery_experts_results = bigquery_experts_query_job.to_dataframe() # Your code goes here



# Preview results

print(bigquery_experts_results.head())



# Check your answer

q_5.check()
#q_5.hint()

#q_5.solution()
def get_experts(topic, client):

    

    experts_query = """SELECT t.owner_user_id as user_id, count(1) as number_of_answers

                    FROM `bigquery-public-data.stackoverflow.posts_answers` t

                    INNER JOIN `bigquery-public-data.stackoverflow.posts_questions` p

                    on t.parent_id = p.id 

                    WHERE p.tags like '%"""+topic+"""%'

                    group by user_id

                    order by number_of_answers desc """

    print("Finding the Top 20 Experts in: "+topic)

    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**11)

    experts_query_job = client.query(experts_query, job_config = safe_config)

    experts_results = experts_query_job.to_dataframe()

    #We're gonna limit the data to 20 results but this can be edited or added to the process!

    return experts_results.head(20)
get_experts("bigquery", client)
q_6.solution()
def expert_finder(topic, client):

    '''

    Returns a DataFrame with the user IDs who have written Stack Overflow answers on a topic.



    Inputs:

        topic: A string with the topic of interest

        client: A Client object that specifies the connection to the Stack Overflow dataset



    Outputs:

        results: A DataFrame with columns for user_id and number_of_answers. Follows similar logic to bigquery_experts_results shown above.

    '''

    my_query = """

               SELECT a.owner_user_id AS user_id, COUNT(1) AS number_of_answers

               FROM `bigquery-public-data.stackoverflow.posts_questions` AS q

               INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a

                   ON q.id = a.parent_Id

               WHERE q.tags like '%{topic}%'

               GROUP BY a.owner_user_id

               """



    # Set up the query (a real service would have good error handling for 

    # queries that scan too much data)

    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)      

    my_query_job = client.query(my_query, job_config=safe_config)



    # API request - run the query, and return a pandas DataFrame

    results = my_query_job.to_dataframe()



    return results
#Why is this returning empty? It's because the query in itself is looking at "topic" as a string and not a variable. 

#I fixed it in my model! 

expert_finder("bigquery", client)