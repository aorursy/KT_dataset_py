# SQql-scavenger-hunt- ARNOLD
# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # import our bq_helper package

from subprocess import check_output
print("Welcome to SQL_Big-Query Hunt Notebook by Arnold")

# create a helper object for our bigquery dataset
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")

print("All the tables in the github_repos dataset")
# print a list of all the tables in the github_repos dataset
github_repos.list_tables()
print("Information on all the columns in the 'commits' table")
# print information on all the columns in the "commits" table in the github_repos dataset
github_repos.table_schema("commits")
# preview the first couple lines of the "commits" table
github_repos.head("commits")
# preview the first ten entries in the by column of the full table
github_repos.head("commits", selected_columns="message", num_rows=10)

#$$$$$$$$$$$$$$$$$$$$ Estimate_Query_Size $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# this query looks in the commit table in the github_repos
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT message
            FROM `bigquery-public-data.github_repos.commits`
            WHERE message = "~fix" """

# check how big this query will be
github_repos.estimate_query_size(query)

# only run this query if it's less than 100 MB
github_repos.query_to_pandas_safe(query, max_gb_scanned=0.1)
import bq_helper # import our bq_helper package

# create a helper object for our bigquery dataset
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
github_repos.head('contents')
#Query to select occurrences of each copies from Content table 
query_copies_counts = """ SELECT license, COUNT(repo_name) as Occurences 
                          FROM `bigquery-public-data.github_repos.licenses`
                          GROUP BY license
                      """
copies_counts = github_repos.query_to_pandas_safe(query_copies_counts)
copies_counts.head()
import bq_helper # import our bq_helper package

# create a helper object for our bigquery dataset
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")

#Number of file per license
# You can use two dashes (--) to add comments in SQL
query_r = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

# check how big this query will be
github_repos.estimate_query_size(query_r)

file_count_by_license = github_repos.query_to_pandas_safe(query_r, max_gb_scanned=6)
print(file_count_by_license)
query = """
WITH python_repos AS (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""

num_commits = github_repos.query_to_pandas_safe(query, max_gb_scanned=10)
print(num_commits)