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
query = ("""
        SELECT sc.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON sf.repo_name = sc.repo_name 
        WHERE sf.path LIKE '%.py'
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """)
commit_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# Your code goes here :)
commit_count_by_license
query = ("""
        WITH python_repos AS (
        SELECT DISTINCT repo_name
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py')
        SELECT sc.repo_name, COUNT(commit) AS num_commits
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        JOIN python_repos
          ON python_repos.repo_name = sc.repo_name
        GROUP BY sc.repo_name
        ORDER BY num_commits DESC
        """)
commit_count_by_license_distinct = github.query_to_pandas_safe(query, max_gb_scanned=6)
commit_count_by_license_distinct
query = ("""
        SELECT repo_name, COUNT(path) AS num_python_files
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py' 
          AND repo_name IN ('torvalds/linux', 'apple/swift', 'Microsoft/vscode', 
          'facebook/react', 'tensorflow/tensorflow')
        GROUP BY repo_name
        ORDER BY num_python_files DESC
        """)
num_files = github.query_to_pandas_safe(query, max_gb_scanned=6)
num_files
