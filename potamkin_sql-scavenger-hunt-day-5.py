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
# How many commits have been made in repos written in Python?

query = """ SELECT COUNT(sc.commit) AS commits, sf.repo_name AS repo_name
            FROM `bigquery-public-data.github_repos.sample_commits` AS sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
                ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'
            GROUP BY repo_name
            ORDER BY commits DESC
"""

com_to_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
com_to_repo

# Thanks to Rachael for hosting the SQL Scavanger Hunt
# I hope everyone enjoyed it aas much as I did