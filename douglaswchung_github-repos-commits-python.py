from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "github_repos" dataset

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "licenses" table

sample_commits_ref = dataset_ref.table("sample_commits")



# API request - fetch the table

sample_commits_table = client.get_table(sample_commits_ref)



# Preview the first five lines of the "licenses" table

client.list_rows(sample_commits_table, max_results=5).to_dataframe()
# Your code here

query = """

        -- Select all the columns we want in our joined table

        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits

        FROM `bigquery-public-data.github_repos.sample_files` as sf

        -- Table to merge into sample_files

        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 

            ON sf.repo_name = sc.repo_name -- what columns should we join on?

        WHERE sf.path LIKE '%.py'

        GROUP BY sf.repo_name

        ORDER BY number_of_commits DESC

        """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

commits_per_repo_python = query_job.to_dataframe()
# Print the dataframe

print(commits_per_repo_python)