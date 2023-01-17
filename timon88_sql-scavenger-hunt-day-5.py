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
import matplotlib.pyplot as plt
query_python = '''SELECT sf.repo_name as Repository, COUNT(commit) AS Commit_counter
                  FROM 
                 `bigquery-public-data.github_repos.sample_files` sf
                  INNER JOIN 
                 `bigquery-public-data.github_repos.sample_commits` sc
                  ON  sf.repo_name = sc.repo_name
                  WHERE path LIKE '%.py'
                  GROUP BY Repository
                  ORDER BY Commit_counter DESC
               '''
Commits_counter = github.query_to_pandas(query_python)
print(Commits_counter)

plt.barh(Commits_counter.Repository, Commits_counter.Commit_counter)
plt.title("Python commits in repos")