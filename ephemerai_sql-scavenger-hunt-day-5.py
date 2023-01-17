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
from bq_helper import BigQueryHelper as bqh

github = bqh(active_project = 'bigquery-public-data',
            dataset_name = 'github_repos')
# all the tables in 'github'
github.list_tables()
github.head("sample_commits")
github.head("sample_files")
# How many commits have been made in repos written in the Python programming language?
query = """
            SELECT sf.repo_name, 
                COUNT(DISTINCT sc.commit) AS commits
            FROM `bigquery-public-data.github_repos.sample_files` AS sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
                ON sf.repo_name = sc.repo_name
            WHERE sf.path LIKE '%.py'
            GROUP BY sf.repo_name
            ORDER BY commits DESC
        """

# Estimate Size
github.estimate_query_size(query)
# Result
commits_in_repos_py = github.query_to_pandas_safe(query, max_gb_scanned = 6)
print(commits_in_repos_py)