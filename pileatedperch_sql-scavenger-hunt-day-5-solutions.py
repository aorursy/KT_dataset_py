import bq_helper
github_repos = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')
query = """
SELECT licenses.license, COUNT(files.id) AS num_files
FROM `bigquery-public-data.github_repos.sample_files` AS files
JOIN `bigquery-public-data.github_repos.licenses` AS licenses -- JOIN is the same as INNER JOIN
    ON licenses.repo_name = files.repo_name
GROUP BY license
ORDER BY num_files DESC
"""

license_counts = github_repos.query_to_pandas_safe(query, max_gb_scanned=10)
license_counts.shape
license_counts
query = """
WITH python_repos AS (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)
query = """
WITH python_repos AS (
    SELECT DISTINCT repo_name -- Notice DISTINCT
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS num_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY num_commits DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)
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
query = """
SELECT repo_name, COUNT(path) AS num_python_files
FROM `bigquery-public-data.github_repos.sample_files`
WHERE repo_name IN ('torvalds/linux', 'apple/swift', 'Microsoft/vscode', 'facebook/react', 'tensorflow/tensorflow')
    AND path LIKE '%.py'
GROUP BY repo_name
ORDER BY num_python_files DESC
"""

github_repos.query_to_pandas_safe(query, max_gb_scanned=10)