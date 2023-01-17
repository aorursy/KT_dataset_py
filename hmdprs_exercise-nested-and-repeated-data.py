# set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql_advanced.ex3 import *

print("Setup is completed")
# create a "Client" object

from google.cloud import bigquery

client = bigquery.Client()



# construct a reference to the "github_repos" dataset

dataset_ref = client.dataset("github_repos", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# construct a reference to the "sample_commits" table

table_ref = dataset_ref.table("sample_commits")



# API request - fetch the table

sample_commits_table = client.get_table(table_ref)



# preview the first five lines of the table

client.list_rows(sample_commits_table, max_results=5).to_dataframe()
# print information on all the columns in the table

sample_commits_table.schema
# write a query to find the answer

max_commits_query = """

SELECT

    committer.name AS committer_name,

    COUNT(*) AS num_commits

FROM

    `bigquery-public-data.github_repos.sample_commits`

WHERE

    committer.date >= '2016-01-01' AND committer.date < '2017-01-01'

GROUP BY

    committer_name

ORDER BY

    num_commits DESC

"""



# check your answer

q_1.check()
# lines below will give you a hint or solution code

# q_1.hint()

# q_1.solution()
# construct a reference to the "languages" table

table_ref = dataset_ref.table("languages")



# API request - fetch the table

languages_table = client.get_table(table_ref)



# preview the first five lines of the table

client.list_rows(languages_table, max_results=5).to_dataframe()
# print information on all the columns in the table

languages_table.schema
# fill in the blank

num_rows = 6



# check your answer

q_2.check()
# lines below will give you a hint or solution code

# q_2.hint()

# q_2.solution()
# write a query to find the answer, use UNNEST always with AS

pop_lang_query = """

SELECT

    language.name as language_name,

    COUNT(*) as num_repos

FROM

    `bigquery-public-data.github_repos.languages`,

    UNNEST(language) AS language

GROUP BY

    language_name

ORDER BY

    num_repos DESC

"""



# check your answer

q_3.check()
# lines below will give you a hint or solution code

# q_3.hint()

# q_3.solution()
# your code here

all_langs_query = """

SELECT

    language.name as name,

    SUM(language.bytes) as bytes

FROM

    `bigquery-public-data.github_repos.languages`,

    UNNEST(language) AS language

WHERE

    repo_name = 'polyrabbit/polyglot'

GROUP BY

    name

ORDER BY

    bytes DESC

"""



# check your answer

q_4.check()
# lines below will give you a hint or solution code

# q_4.hint()

# q_4.solution()