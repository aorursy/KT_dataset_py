import bq_helper


github = bq_helper.BigQueryHelper(
    active_project="bigquery-public-data",
    dataset_name="github_repos"
)
query = """
    SELECT sc.repo_name, COUNT(sc.commit) as count
    FROM `bigquery-public-data.github_repos.sample_files` as sf
    INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc ON sc.repo_name = sf.repo_name
    WHERE sf.path like '%.py'
    GROUP BY sc.repo_name
    ORDER BY count DESC
"""
result = github.query_to_pandas(query)
result