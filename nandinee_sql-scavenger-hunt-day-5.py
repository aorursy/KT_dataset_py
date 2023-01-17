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
## How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
## You'll want to JOIN the sample_files and sample_commits questions to answer this.
## Hint:You can figure out which files are written in Python by filtering results from the 
## "sample_files" table using `WHERE path LIKE '%.py'`.
## This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.

query = ("""
        -- Select all the columns we want in our joined table
        SELECT sf.repo_name, COUNT(c.commit) AS number_of_commit
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as c 
            ON sf.repo_name = c.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commit DESC
        """)
file_count_by_commit = github.query_to_pandas_safe(query, max_gb_scanned=7)
print(file_count_by_commit)
