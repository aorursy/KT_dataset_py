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

    SELECT committer.name AS committer_name, COUNT(*) AS num_commits

    FROM `bigquery-public-data.github_repos.sample_commits`

    WHERE committer.date >= '2016-01-01' AND committer.date < '2017-01-01'

    GROUP BY committer.name

    ORDER BY num_commits DESC

                    """



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

SELECT l.name AS language_name, COUNT(*) AS num_repos

FROM `bigquery-public-data.github_repos.languages`,

UNNEST(language) AS l

GROUP BY language_name

ORDER BY num_repos DESC



                 """



# Check your answer

q_3.check()
# Lines below will give you a hint or solution code

#q_3.hint()

#q_3.solution()
# Your code here

all_langs_query = """

SELECT l.name, l.bytes

FROM `bigquery-public-data.github_repos.languages`,

UNNEST(language) AS l

WHERE repo_name = 'polyrabbit/polyglot'

ORDER BY l.bytes DESC

                  """



# Check your answer

q_4.check()
# Lines below will give you a hint or solution code

#q_4.hint()

#q_4.solution()