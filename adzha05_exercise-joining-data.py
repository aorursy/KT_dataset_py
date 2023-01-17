# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("sample_commits")
github.head("sample_files")
query = """SELECT sf.repo_name, COUNT(c.commit) as commit_amount
FROM `bigquery-public-data.github_repos.sample_files` as sf
INNER JOIN `bigquery-public-data.github_repos.sample_commits` as c
    ON  sf.repo_name = c.repo_name
WHERE path LIKE '%.py'
GROUP BY sf.repo_name
ORDER BY commit_amount DESC
"""
commit_per_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(commit_per_repo)