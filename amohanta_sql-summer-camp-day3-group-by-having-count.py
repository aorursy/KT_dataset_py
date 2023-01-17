from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "hacker_news" dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there are four !)

for table in tables:

    print(table.table_id)
# Construct a reference to the "comments" table

table_ref = dataset_ref.table("comments")



# API request - fetch the table

table = client.get_table(table_ref)



# Print information on all columns in the "comments" table in the "hacker_news" dataset

table.schema
# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
t1=client.list_rows(table, max_results=50000).to_dataframe()

t1.shape

from collections import Counter

l1=list(t1.loc[:,'parent'])

for key, value in Counter(l1).items():

    if key == 1492475:

        print('Yes')
# Query to select comments that received more than 10 replies

query_popular = """

                SELECT parent, COUNT(id)

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

                """
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_popular, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



# Print the first five rows of the DataFrame

popular_comments.head()
# Improved version of earlier query, now with aliasing & improved readability

query_improved = """

                 SELECT parent, COUNT(1) AS NumPosts

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY parent

                 HAVING COUNT(1) > 10

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()



# rows will appear in different order in each time
# Improved version of earlier query, now with aliasing & improved readability

query_improved = """

                 SELECT parent, COUNT(1) AS NumPosts

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY parent

                 HAVING COUNT(1) > 10

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()



# rows will appear in different order in each time
# Improved version of earlier query, now with aliasing & improved readability

query_improved = """

                 SELECT parent, COUNT(*) AS NumPosts

                 FROM `bigquery-public-data.hacker_news.comments`

                 GROUP BY parent

                 HAVING COUNT(*) > 10

                 """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_improved, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

improved_df = query_job.to_dataframe()



# Print the first five rows of the DataFrame

improved_df.head()



# rows will appear in different order in each time