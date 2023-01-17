# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """SELECT f.repo_name, COUNT(sc.commit)
            FROM `bigquery-public-data.github_repos.files` AS f
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc ON f.repo_name = sc.repo_name
            WHERE f.path LIKE '%.py' 
            GROUP BY repo_name 
            ORDER BY COUNT(sc.commit) DESC 
            LIMIT 10
"""
#github.estimate_query_size(query) <-- 1.6GB
py_commits = github.query_to_pandas_safe(query, max_gb_scanned=2)
print(py_commits)