# import package with helper functions 
import bq_helper
github_repos = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')
query = """
SELECT sf.repo_name as repo, COUNT(DISTINCT sc.commit) AS commits
FROM `bigquery-public-data.github_repos.sample_commits` as sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
    ON sf.repo_name = sc.repo_name
WHERE sf.path LIKE '%.py'
GROUP BY repo
ORDER BY commits DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)

