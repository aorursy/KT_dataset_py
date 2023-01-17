# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
# Your code goes here :)
# Query 1: How many commits (recorded in the "sample_commits" table) have been made in 
# repos written in the Python programming language?
# You can use two dashes (--) to add comments in SQL
# using DISTINCT to avoid duplication
query1 = ("""
        -- Select all the columns we want in our joined table
        SELECT sc.repo_name, COUNT(DISTINCT sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """)
# check how big this query will be (using estimate_query_size())
github.estimate_query_size(query1)
# do query1
total_number_commits_in_python = github.query_to_pandas_safe(query1, max_gb_scanned=6)
# display results
total_number_commits_in_python