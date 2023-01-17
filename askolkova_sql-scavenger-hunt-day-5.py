# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
# Your code goes here :)
github.head("sample_files")

github.head("sample_commits")
query1 = ("""
        SELECT sf.repo_name AS repo, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY repo
        ORDER BY number_of_commits DESC
        """)

commits_count_by_repo = github.query_to_pandas_safe(query1, max_gb_scanned=9)
commits_count_by_repo
import matplotlib.pyplot as plt
import numpy as np

plt.barh(np.arange(len(commits_count_by_repo.repo)), commits_count_by_repo.number_of_commits, align='center', alpha=0.5)
plt.yticks(np.arange(len(commits_count_by_repo.repo)), commits_count_by_repo.repo)
plt.xlabel('Commits')
plt.title('Python commits by Repo')