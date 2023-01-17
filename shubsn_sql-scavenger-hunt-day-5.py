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
myquery = ("""
        -- Select all the columns we want in our joined table
        SELECT commits.repo_name, COUNT(commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as commits
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as S 
            ON S.repo_name = commits.repo_name -- what columns should we join on?
            WHERE S.path LIKE '%.py'
        GROUP BY commits.repo_name
        ORDER BY number_of_commits DESC
        """)

num_commits_Py_files = github.query_to_pandas_safe(myquery, max_gb_scanned=6)

print(num_commits_Py_files)
