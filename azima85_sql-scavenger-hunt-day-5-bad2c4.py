# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_files').columns
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
file_count_by_license
# print out all the returned results
#print(file_count_by_license)
github.head('sample_commits')
# Your code goes here :)
query1="""WITH python AS (SELECT repo_name FROM `bigquery-public-data.github_repos.sample_files` 
                        WHERE path LIKE '%.py')
                SELECT python.repo_name AS repo, COUNT(sc.commit) AS no_of_commits
                FROM python
                INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
                ON sc.repo_name=python.repo_name
                GROUP BY repo"""
commits_by_repo=github.query_to_pandas_safe(query1,max_gb_scanned=6)
commits_by_repo