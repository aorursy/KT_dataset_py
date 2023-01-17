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

#Select the columns author and apply a COUNT to the ID column

#Group by "author" to calculate the final number of post per author

#Add the condition that you only are interested in those results where the number of post are equal or greater than 10k









prolific_commenters_query ="""

                            SELECT author,COUNT(ID) AS NumPosts

                            FROM `bigquery-public-data.hacker_news.comments`

                            GROUP BY author

                            HAVING COUNT(ID) >= 10000

                            """ # Your code goes here



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



# Select the columns deleted and ID. Apply a count to the ID column to get the number of deleted posts

# To get the answer it is enough with GROUP the results by the column deleted.

# The dataframe will display the number of registers where the column deleted is equal to True and None.





deleted_comments_query ="""

                            SELECT deleted, COUNT(ID) AS NumDeletedPosts

                            FROM `bigquery-public-data.hacker_news.comments`

                            GROUP BY deleted

                            """ # Your code goes here



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(deleted_comments_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

deleted_comments = query_job.to_dataframe()



# View top few rows of results

print(deleted_comments.head())



num_deleted_posts = 227736 # Put your answer here



# Check your answer

q_2.check()
#q_2.solution()