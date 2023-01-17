import pandas as pd
import bq_helper

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                dataset_name="github_repos")
github.list_tables()
github.head('sample_files')
github.head('sample_commits')
q = """
SELECT count(sc.commit) AS commit_count , sf.repo_name AS repository 
FROM `bigquery-public-data.github_repos.sample_commits` AS sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
ON sc.repo_name = sf.repo_name
WHERE sf.path LIKE '%.py'
GROUP BY repository
ORDER BY commit_count DESC
"""
github.estimate_query_size(q)
python_rep = github.query_to_pandas_safe(q, max_gb_scanned=6)
python_rep.head()