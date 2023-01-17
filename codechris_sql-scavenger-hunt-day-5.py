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
query = """ SELECT COUNT(c.commit) as commits
            FROM `bigquery-public-data.github_repos.sample_commits` as c
            INNER JOIN `bigquery-public-data.github_repos.sample_files` as f
                ON c.repo_name = f.repo_name
            WHERE f.path LIKE '%.py'
            ORDER BY COUNT(c.commit) DESC
            """


py_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
py_commits