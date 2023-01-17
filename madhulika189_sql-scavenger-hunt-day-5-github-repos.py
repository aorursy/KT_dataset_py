import pandas as pd
import bq_helper
github_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="github_repos")

# Listing tables
print(github_data.list_tables())
# Listing schema of the tables to be used
print("sample_commits:\n\n")
print(github_data.table_schema("sample_commits"))

print("\n\n sample_files:")
print(github_data.table_schema("sample_files"))
query = """
SELECT 
    c.repo_name,
    COUNT(c.commit) AS Num_Commits
FROM `bigquery-public-data.github_repos.sample_files` AS s 
INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS C 
ON s.repo_name = c.repo_name
WHERE s.path LIKE '%.py%'
GROUP BY c.repo_name
ORDER BY Num_Commits DESC
"""

py_commits = github_data.query_to_pandas_safe(query,max_gb_scanned=6)
print("Top 10 repos by number of commits in Python:")
py_commits.head(n=10)