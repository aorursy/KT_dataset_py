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
github.head("sample_commits",3)
github.head("sample_files",3)
query = """
SELECT COUNT(comm.commit) AS numer_of_commits, file.repo_name AS name
FROM `bigquery-public-data.github_repos.sample_commits` AS comm
INNER JOIN `bigquery-public-data.github_repos.sample_files` AS file
ON comm.repo_name = file.repo_name
WHERE file.path LIKE "%.py"
GROUP BY file.repo_name
"""
#query = """
#SELECT COUNT(comm.commit)
#FROM `bigquery-public-data.github_repos.sample_commits` AS comm
#"""
test = github.query_to_pandas_safe(query, max_gb_scanned=6)
test