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
# How many commits (recorded in the "sample_commits" table) 
# have been made in repos written in the Python programming language? 
# (I'm looking for the number of commits per repo for all the repos written in Python.

query2 = ("""
        -- Select all the columns we want in our joined table
        SELECT COUNT(sc.commit) AS number_of_commits, sc.repo_name --What to display
        FROM `bigquery-public-data.github_repos.sample_commits` as sc --Where to pull info
        
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf  -- joining datasets
            ON sc.repo_name = sf.repo_name -- what columns should we join on?
            WHERE path LIKE '%.py' --Only python commits
        GROUP BY sc.repo_name --group by rep
        ORDER BY number_of_commits DESC --order by number of python commits
        """)

python_commits_per_repo = github.query_to_pandas_safe(query2, max_gb_scanned=6)


print(python_commits_per_repo)