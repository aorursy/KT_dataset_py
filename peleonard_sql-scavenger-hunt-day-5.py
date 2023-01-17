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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github
github.head("sample_commits")
github.head("sample_files")
query = ("""WITH py_repos AS (
                SELECT DISTINCT repo_name 
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%py'
            )
                
            --find number of commits made in py_repos
            SELECT commits.repo_name, COUNT(commits.commit) as number_of_commits
            FROM `bigquery-public-data.github_repos.sample_commits` as commits
            INNER JOIN py_repos ON py_repos.repo_name = commits.repo_name
            GROUP BY repo_name
            ORDER BY number_of_commits DESC
         """)
github.query_to_pandas_safe(query, max_gb_scanned=20)
# Get the count of each value in repo_name of sample_commits
query_sc = """SELECT DISTINCT repo_name, COUNT(repo_name) AS numRepoName
              FROM `bigquery-public-data.github_repos.sample_commits`
              GROUP BY repo_name"""
github.query_to_pandas_safe(query_sc, max_gb_scanned=20)
# the repo_name in sample commits is not distinct
# Get the count of each value in repo_name of sample_files and the list the first 10 count
query_sf = """SELECT DISTINCT repo_name, COUNT(repo_name) AS numRepoName
            FROM `bigquery-public-data.github_repos.sample_files`
            GROUP BY repo_name
            ORDER BY numRepoName DESC
            LIMIT 10"""
github.query_to_pandas_safe(query_sf, max_gb_scanned=20)
query_ij = """SELECT sf.path AS path, sf.repo_name AS repoName, sc.commit as commit
            FROM `bigquery-public-data.github_repos.sample_files` AS sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
                ON sf.repo_name = sc.repo_name"""
github.query_to_pandas_safe(query_ij, max_gb_scanned=20)