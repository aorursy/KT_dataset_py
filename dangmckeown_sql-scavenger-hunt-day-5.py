# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

query = """SELECT COUNT(sample_commits.commit) AS count_commits, sample_commits.repo_name AS python_repos
        FROM `bigquery-public-data.github_repos.sample_commits` AS sample_commits
        JOIN `bigquery-public-data.github_repos.sample_files` AS sample_files
        ON sample_commits.repo_name = sample_files.repo_name
        WHERE sample_files.path LIKE '%.py'
        GROUP BY python_repos
        ORDER BY count_commits DESC
"""
python_commits = github.query_to_pandas(query)

python_commits

