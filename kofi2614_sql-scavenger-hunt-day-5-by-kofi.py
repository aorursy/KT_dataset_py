# How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language?
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query_1 = """
        SELECT b.Repo_name, COUNT(b.commit) AS Commits_Python
        FROM `bigquery-public-data.github_repos.sample_files` as a
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as b 
        ON a.repo_name = b.repo_name 
        WHERE path LIKE '%.py'
        GROUP BY b.repo_name
        ORDER BY Commits_Python
        """
counts = github.query_to_pandas(query_1)
counts