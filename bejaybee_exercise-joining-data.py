# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# Your code here
query = ("""
        -- Select all the columns we want in our joined table
        SELECT sf.repo_name, COUNT(sl.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
                -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sl 
        ON sf.repo_name = sl.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)
file_count_by_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(file_count_by_commits )
#github.head("sample_files")