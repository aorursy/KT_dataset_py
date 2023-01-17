# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query1 = ("""
        -- Select all the columns we want in our joined table
        SELECT sc.repo_name as Repo, COUNT(DISTINCT sc.commit) AS Number_Of_Commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY Repo
        ORDER BY Number_Of_Commits DESC
        """)

file_Commit_Count_By_Repo = github.query_to_pandas_safe(query1, max_gb_scanned=6)
# print out all the returned results
print(file_Commit_Count_By_Repo)