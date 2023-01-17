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
# BigQuery default JOIN behavior is INNER JOIN
query_num_commits_python = """SELECT count(sc.commit) AS num_commits, sc.repo_name
                                FROM `bigquery-public-data.github_repos.sample_commits` AS sc
                                JOIN `bigquery-public-data.github_repos.sample_files` AS sf
                                ON sc.repo_name = sf.repo_name
                                WHERE sf.path LIKE '%.py'
                                GROUP BY sc.repo_name
                                ORDER BY num_commits DESC
"""
github.estimate_query_size(query_num_commits_python)
num_commits_python = github.query_to_pandas_safe(query_num_commits_python, max_gb_scanned=6)
num_commits_python.head()