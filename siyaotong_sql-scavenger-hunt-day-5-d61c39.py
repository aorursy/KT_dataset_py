import bq_helper
# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        SELECT L.commit, COUNT(L.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as L 
        WHERE path LIKE '%.py'
            ON sf.repo_name = L.repo_name
        GROUP BY L.path
        ORDER BY number_of_commits DESC
        """)
commit_count_in_python = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(commit_count_in_python)