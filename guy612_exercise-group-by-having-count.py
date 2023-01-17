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
# Query to select prolific commenters and post counts

prolific_commenters_query = """

select author,count(id) as NumPosts

from `bigquery-public-data.hacker_news.comments`

group by author

having count(id)>10000

"""# Your code goes here



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(prolific_commenters_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

prolific_commenters = query_job.to_dataframe()



# View top few rows of results

print(prolific_commenters.head())



# Check your answer

q_1.check()
#q_1.solution()
# Write your query here and figure out the answer

query1="""

select count(deleted)

from `bigquery-public-data.hacker_news.comments`



"""

any_variable=bigquery.QueryJobConfig(max=10**10)

job=client.query(query1,job_config=any_variable)

answer=job.to_dataframe()

print(answer)
num_deleted_posts = 227736 # Put your answer here



q_2.check()
#q_2.solution()