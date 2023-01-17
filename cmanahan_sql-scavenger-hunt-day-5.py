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

# looking at each table:
github.head("sample_files")
github.head("sample_commits")
query = ("""
        SELECT 
            sc.repo_name as repos
            ,count(sc.commit) as commit_cnt
           FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN (select repo_name   
                    FROM `bigquery-public-data.github_repos.sample_files` 
                    WHERE path like '%.py' ) sf
            ON sc.repo_name = sf.repo_name 
        GROUP BY ROLLUP(repos)
        ORDER BY commit_cnt DESC
        """)

repo_commit_counts = github.query_to_pandas_safe(query, max_gb_scanned=7)

repo_commit_counts
#  None is the total count from Rollup feature.