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
# My code goes here :) 
# I did not use DISCTINCT COUNT for my initial attempt

query1 = """SELECT c.repo_name, COUNT(c.commit) number_of_commits_py  
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            INNER JOIN `bigquery-public-data.github_repos.sample_files` AS f
            ON c.repo_name = f.repo_name
            WHERE f.path LIKE '%.py'
            GROUP BY c.repo_name
            ORDER BY number_of_commits_py DESC
         """

commit_count_python = github.query_to_pandas_safe(query1, max_gb_scanned = 6)
commit_count_python
# Another way of approaching this problem is 
#by first filtering the "sample_files" table to find which repos had .py files:
# (used SELECT DISTINCT because one repo might have many python files associated with it)
query2 = """WITH python_files AS (
                SELECT DISTINCT repo_name 
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
                )
            SELECT c.repo_name, COUNT(commit) number_of_commits_py
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            INNER JOIN python_files AS pf
            ON c.repo_name = pf.repo_name
            GROUP BY c.repo_name
            ORDER BY number_of_commits_py DESC
        """
commit_count_python_2 = github.query_to_pandas_safe(query2, max_gb_scanned = 6)
commit_count_python_2
# the following query checks the total count of commits for each repo:
query3 = """SELECT repo_name, COUNT(commit) number_of_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            GROUP BY repo_name
            ORDER BY number_of_commits DESC
        """
commit_count_all = github.query_to_pandas_safe(query3, max_gb_scanned = 6)
# You can see that repo "twbs/bootstrap" did not have any .py files
commit_count_all
# I am curious and just want to see how many files had python codes for each repo:
query4 = """SELECT repo_name, COUNT(path) number_of_files_py
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py'
            GROUP BY repo_name
            ORDER BY number_of_files_py DESC
        """
file_count_py = github.query_to_pandas_safe(query4, max_gb_scanned = 6)
# There were 32908 repos that actually had python files! Wow!
file_count_py