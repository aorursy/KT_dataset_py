# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
#Query to count number of commits in python by joining sample_files and sample_commits tables
commit_query = (""" SELECT COUNT(sc.commit) AS commits
                    FROM `bigquery-public-data.github_repos.sample_files` as sf
                    JOIN  `bigquery-public-data.github_repos.sample_commits` as sc
                    ON sf.repo_name = sc.repo_name
                    WHERE path LIKE '%.py'
                """)
commits_in_python = github.query_to_pandas_safe(commit_query, max_gb_scanned=6)
commits_in_python
print('Number of commits written in Python: \n')
print(commits_in_python.commits[0])