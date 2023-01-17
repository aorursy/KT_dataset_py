# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""

        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name 
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
# Your code goes here :)

query2 = ("""
        WITH comm_python AS
        (SELECT sc.commit, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 
            ON sf.repo_name = sc.repo_name 
        WHERE sf.path LIKE '%.py'
        GROUP BY sc.commit)
        SELECT SUM(number_of_files)
        FROM comm_python
        """)

total_commits = github.query_to_pandas_safe(query2, max_gb_scanned=6)
print(total_commits)
