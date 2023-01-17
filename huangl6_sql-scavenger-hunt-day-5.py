# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = """WITH chunk AS(
        SELECT sf.path AS path, sc.commit AS commit, sc.repo_name AS repo
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE path LIKE '%.py'
        )
        SELECT COUNT(commit) AS numCommits, repo
        FROM chunk
        GROUP BY repo
        ORDER BY numCommits DESC
        """

pythonCountPerRepo = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(pythonCountPerRepo)
#finding total number of python commits
query1 = """WITH chunk AS(
        SELECT sf.path AS path, sc.commit AS commit, sc.repo_name AS repo
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE path LIKE '%.py'
        )
        SELECT COUNT(commit) AS numCommits
        FROM chunk
        """
pythonCount = github.query_to_pandas_safe(query1, max_gb_scanned=6)
# print out all the returned results
print(pythonCount)