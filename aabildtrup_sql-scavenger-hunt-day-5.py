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
github.head("sample_files")
github.head("sample_commits")
# Your code goes here :)
#Inner query: repos containing .py files.
#Outer query: Number of commits to repositories containing .py files.
query2 = """WITH reps AS
            (
                SELECT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
                GROUP BY repo_name
            )
            SELECT r.repo_name, COUNT(sc.commit) AS commit_count
            FROM reps as r
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc ON r.repo_name  = sc.repo_name
            GROUP BY r.repo_name
            ORDER BY COUNT(sc.commit)
         """
github.estimate_query_size(query2)
py_commits = github.query_to_pandas_safe(query2, max_gb_scanned = 6)
py_commits