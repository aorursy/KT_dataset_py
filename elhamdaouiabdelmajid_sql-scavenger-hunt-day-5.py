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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = (""" WITH py_repos AS 
                (
                    SELECT repo_name
                    FROM `bigquery-public-data.github_repos.sample_files`
                    WHERE path LIKE '%.py'
                )
            SELECT C.repo_name, COUNT(C.commit) AS nbr_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS C
            JOIN py_repos as PR
            ON  PR.repo_name = C.repo_name
            GROUP BY C.repo_name
            ORDER BY nbr_commits DESC
        """)

commits_py_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
commits_py_repo.head(8)