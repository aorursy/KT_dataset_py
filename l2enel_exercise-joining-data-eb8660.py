# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_commits', 3)
github.head('sample_files', 3)
query = """SELECT COUNT(commit) as num_commits
    FROM `bigquery-public-data.github_repos.sample_commits` AS sc
    JOIN `bigquery-public-data.github_repos.sample_files` AS sf ON sc.repo_name = sf.repo_name
    WHERE sf.path LIKE '%.py%'
"""

#github.estimate_query_size(query)
github.query_to_pandas(query)