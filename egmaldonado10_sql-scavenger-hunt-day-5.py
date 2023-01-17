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

query2 = ("""
        SELECT COUNT(commit) AS py_commits
        FROM `bigquery-public-data.github_repos.sample_commits` AS sample_commits
        INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sample_files 
            ON sample_commits.repo_name = sample_files.repo_name
            WHERE sample_files.path LIKE '%.py'        
        """)

sample_commits_py = github.query_to_pandas_safe(query2, max_gb_scanned=6)
sample_commits_py