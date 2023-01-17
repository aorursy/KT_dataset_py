from google.cloud import bigquery
# Create a "Client" object

client = bigquery.Client()
# Construct a reference to the "hacker_news" dataset

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "comments" table

table_ref = dataset_ref.table("commits")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "comments" table

client.list_rows(table, max_results=5).to_dataframe()
table.schema
# Query to select comments that received more than 10 replies

query_popular = """

                SELECT tree, 

                    COUNT(1) as commit_number

                FROM `bigquery-public-data.github_repos.commits`

                WHERE difference_truncated IS NOT NULL

                GROUP BY tree

                """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query_popular, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

popular_comments = query_job.to_dataframe()



# Print the first five rows of the DataFrame

popular_comments.head()