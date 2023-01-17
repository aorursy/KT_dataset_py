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
query = ("""
        SELECT COUNT(sc.commit) AS number_python_commits,
        sc.repo_name
        FROM `bigquery-public-data.github_repos.sample_commits` sc
        JOIN `bigquery-public-data.github_repos.sample_files` sf
        ON sc.repo_name = sf.repo_name
        WHERE sf.path like '%.py'
        GROUP BY sc.repo_name
        ORDER BY COUNT(sc.commit) DESC
        """)
# github.estimate_query_size(query)
number_python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(number_python_commits)
# Your code goes here :)
query = ("""
        WITH distinct_repos AS (
            SELECT DISTINCT repo_name
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path like '%.py'
        )
        SELECT COUNT(sc.commit) AS number_python_commits,
        sc.repo_name
        FROM `bigquery-public-data.github_repos.sample_commits` sc
        JOIN distinct_repos dr
        ON sc.repo_name = dr.repo_name        
        GROUP BY sc.repo_name
        ORDER BY COUNT(sc.commit) DESC
        """)
github.estimate_query_size(query)
number_python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(number_python_commits)