
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

query1 = ("""
        SELECT F.repo_name, COUNT(C.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as C
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as F 
            ON F.repo_name = C.repo_name
        GROUP BY repo_name
        ORDER BY number_of_commits DESC
        """)
        
commits_count_by_repo = github.query_to_pandas_safe(query1, max_gb_scanned=6)
print(commits_count_by_repo)