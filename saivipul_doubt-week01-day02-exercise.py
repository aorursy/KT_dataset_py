#GOTO FOURTH In[] 

#FOURTH IN[] IS ACCORDING TO SOLUTION

#FIFTH IN[] IS WHAT I HAVE DONE

#WHY IS THERE ERROR

#IT ALSO SAYS FIFTH IN[] INCORRECT

#ONLY DIFFERENCE IS I USED __WHERE__ AND IN THE SOLUTION USED __HAVING__ 
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



tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)



# Construct a reference to the "comments" table

table_ref = dataset_ref.table("comments")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to select prolific commenters and post counts

prolific_commenters_query = """

        SELECT author, COUNT(1) AS NumPosts

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY author

        HAVING COUNT(1) > 10000

        """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10*9)

query_job = client.query(prolific_commenters_query, job_config=safe_config)

prolific_commenters = query_job.to_dataframe()

print(prolific_commenters.head())

q_1.check()
prolific_commenters_query = """

        SELECT author, COUNT(1) AS NumPosts

        FROM `bigquery-public-data.hacker_news.comments`

        GROUP BY author

        WHERE COUNT(1) > 10000

        """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10*9)

query_job = client.query(prolific_commenters_query, job_config=safe_config)

prolific_commenters = query_job.to_dataframe()

print(prolific_commenters.head())

q_1.check()