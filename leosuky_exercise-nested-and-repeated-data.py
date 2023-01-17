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

                     SELECT committer.name as committer_name, COUNT(committer.date) as num_commits

                     FROM `bigquery-public-data.github_repos.sample_commits`

                     WHERE EXTRACT(YEAR FROM committer.date) = 2016

                     GROUP BY committer_name

                     ORDER BY num_commits DESC

                    """



max_commits_result = client.query(max_commits_query).result().to_dataframe()



max_commits_result.head()



# Check your answer

q_1.check()
# Lines below will give you a hint or solution code

#q_1.hint()

#q_1.solution()
# Construct a reference to the "languages" table

table_ref = dataset_ref.table("languages")



# API request - fetch the table

languages_table = client.get_table(table_ref)



# Preview the first five lines of the table

client.list_rows(languages_table, max_results=5).to_dataframe()
# Print information on all the columns in the table

languages_table.schema
# Fill in the blank

num_rows = 6



# Check your answer

q_2.check()
# Lines below will give you a hint or solution code

#q_2.hint()

#q_2.solution()
# Write a query to find the answer

pop_lang_query = """

                 SELECT lg.name AS language_name, COUNT(repo_name) AS num_repos

                 FROM `bigquery-public-data.github_repos.languages`,

                 UNNEST(language) AS lg

                 GROUP BY language_name

                 ORDER BY num_repos DESC

                 """



pop_lang_result = client.query(pop_lang_query).result().to_dataframe()



pop_lang_result.head()



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Your code here

all_langs_query = """

                 SELECT lg.name AS name, lg.bytes AS bytes

                 FROM `bigquery-public-data.github_repos.languages`,

                 UNNEST(language) AS lg

                 WHERE repo_name LIKE '%polyrabbit/polyglot%'

                 ORDER BY bytes DESC

                  """

all_langs_result = client.query(all_langs_query).result().to_dataframe()

all_langs_result.head()



# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()