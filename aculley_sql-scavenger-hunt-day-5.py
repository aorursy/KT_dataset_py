# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_files')
github.head('licenses')
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
github.head('sample_commits')
query = """SELECT repo_name, COUNT(path)
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            WHERE sf.path LIKE '%.py'
            GROUP BY repo_name
            ORDER BY COUNT(path) desc
        """

file_count_by_repo_name = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(file_count_by_repo_name.head())
len(file_count_by_repo_name)
query = """SELECT repo_name, COUNT(commit) as commit_count
            FROM `bigquery-public-data.github_repos.sample_commits`
            GROUP BY repo_name
        """

commits_by_repo_name = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(commits_by_repo_name.head())
len(commits_by_repo_name)
query = """
        WITH sf AS 
        (
            SELECT DISTINCT repo_name
            FROM `bigquery-public-data.github_repos.sample_files` 
            WHERE path LIKE '%.py'
        )
        SELECT L.repo_name, COUNT(L.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as L 
        INNER JOIN sf
            ON sf.repo_name = L.repo_name 
       
        GROUP BY L.repo_name
        ORDER BY number_of_commits DESC
        """

python_commit_count_by_repo_name = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(python_commit_count_by_repo_name)