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
print("How many commits have been made in repos written in the Python programming language")
query = ("""
        SELECT COUNT(commits_table.commit) AS number_of_commits_in_python
        FROM `bigquery-public-data.github_repos.sample_files` as files_table
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as commits_table
            ON files_table.repo_name = commits_table.repo_name 
        WHERE files_table.path LIKE '%.py'
        """)

commits_in_python = github.query_to_pandas_safe(query, max_gb_scanned=6)
commits_in_python