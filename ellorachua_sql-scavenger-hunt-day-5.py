# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query1 = ("""
        -- Select all the columns we want in our joined table
        SELECT C.repo_name, COUNT(path) AS number_of_python_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as C 
            ON sf.repo_name = C.repo_name
            WHERE path LIKE '%.py'
        GROUP BY C.repo_name
        ORDER BY number_of_python_commits DESC
        """)
github.estimate_query_size(query1)
py_commits = github.query_to_pandas_safe(query1, max_gb_scanned=6)
# print out all the returned results
py_commits