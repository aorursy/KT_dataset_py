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
query1 = ("""
        SELECT s_c.repo_name AS RepoName, COUNT(s_f.path) AS FilesQuantity_py
        FROM `bigquery-public-data.github_repos.sample_files` as s_f
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as s_c 
            ON s_f.repo_name = s_c.repo_name
        WHERE s_f.path LIKE '%.py'
        GROUP BY RepoName
        ORDER BY FilesQuantity_py DESC
        """)

py_file_count_by_repo_name = github.query_to_pandas_safe(query1, max_gb_scanned=6)
print(py_file_count_by_repo_name)
py_file_count_by_repo_name.to_csv("py_file_count_by_repo_name_UseGithubReposDataset.csv")