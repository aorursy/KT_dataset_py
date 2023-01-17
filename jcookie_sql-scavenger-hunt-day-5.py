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
# Number of commits per repor or all the GITHUB repos written in Python, samples_files and samples_content

query5= """with temp AS
             (SELECT sf.path AS path, sc.commit AS commit, sc.repo_name AS Repository_Name
              FROM `bigquery-public-data.github_repos.sample_files` AS sf
              INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
              ON sf.repo_name = sc.repo_name
              WHERE path LIKE '%.py'
            )
            SELECT count(commit) AS Number_of_Commits_in_Python, Repository_Name
            FROM temp
            GROUP BY Repository_Name 
            ORDER BY Number_of_Commits_in_Python DESC
            """
result5 = github.query_to_pandas(query5)
result5
