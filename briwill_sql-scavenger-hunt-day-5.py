# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# You can use two dashes (--) to add comments in SQL
query = ("""
        SELECT sf.repo_name AS repo, COUNT(cts.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS cts
            ON sf.repo_name = cts.repo_name 
        WHERE sf.path LIKE '%.py'  
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

commit_count_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
print(commit_count_by_repo)