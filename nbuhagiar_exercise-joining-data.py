# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """SELECT f.repo_name, COUNT(c.commit) as num_commits 
           FROM `bigquery-public-data.github_repos.sample_files` as f 
           INNER JOIN `bigquery-public-data.github_repos.sample_commits` as c 
                      on f.repo_name = c.repo_name
           WHERE f.path LIKE '%.py' 
           GROUP BY f.repo_name"""
df = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(df)
