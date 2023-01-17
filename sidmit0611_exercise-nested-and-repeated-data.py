# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql_advanced.ex3 import *
print("Setup Complete")
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "github_repos" dataset
dataset_ref = client.dataset("github_repos", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "sample_commits" table
table_ref = dataset_ref.table("sample_commits")

# API request - fetch the table
sample_commits_table = client.get_table(table_ref)

# Preview the first five lines of the table
client.list_rows(sample_commits_table, max_results=5).to_dataframe()
# Print information on all the columns in the table
sample_commits_table.schema
# Write a query to find the answer
max_commits_query = """
                        SELECT committer.name AS committer_name, count(*) AS num_commits
                        FROM `bigquery-public-data.github_repos.sample_commits`
                        WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'
                        GROUP BY committer_name
                        ORDER BY num_commits DESC
                    """

# Construct a reference to the "languages" table
table_ref = dataset_ref.table("languages")

# API request - fetch the table
languages_table = client.get_table(table_ref)

# Preview the first five lines of the table
client.list_rows(languages_table, max_results=5).to_dataframe()
# this is the language table
# Print information on all the columns in the table
languages_table.schema
num_rows = 6
# Write a query to find the answer
pop_lang_query = """
                    SELECT language.name AS language_name, COUNT(*) AS num_repos
                    FROM `bigquery-public-data.github_repos.languages`,
                    UNNEST(language) AS language
                    GROUP BY language_name
                    ORDER BY num_repos DESC
                 """

query = """
        SELECT repo_name, COUNT(language.name) AS Lang
        FROM `bigquery-public-data.github_repos.languages`,
        UNNEST(language) AS language
        GROUP BY repo_name
        ORDER BY Lang DESC
        """

query_job = client.query(query)

repo_with_most_language = query_job.to_dataframe()

repo_with_most_language.head()
# from the above query we can determine that 'polyrabbit/polyglot' is the repo in which most languages is used. as we can see that 
## 216 languages are used in this repo therefore we will be using this repo for finding the result
# Your code here
all_langs_query = """
                     SELECT language.name AS name, language.bytes AS bytes
                     FROM `bigquery-public-data.github_repos.languages`, 
                     UNNEST(language) as language
                     WHERE repo_name LIKE 'polyrabbit/polyglot'
                     ORDER BY bytes DESC
                  """