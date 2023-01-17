import bq_helper
git_hub = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="github_repos")

query = ("""
        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 
            ON sf.repo_name = sc.repo_name
        WHERE (sf.path LIKE '%.py' or 
               sf.path LIKE '%.pyw' or 
               sf.path LIKE '%.pyc' or 
               sf.path LIKE '%.pyo' or 
               sf.path LIKE '%.pyd')    
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

repos_written_in_thePython  = git_hub.query_to_pandas_safe(query, max_gb_scanned=6)
print(repos_written_in_thePython)