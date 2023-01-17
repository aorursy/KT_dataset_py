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
# print the first couple rows of the "sample_commits" table
github.head("sample_commits")
# You can use two dashes (--) to add comments in SQL
query_commit = ("""
        WITH repolist AS 
            (
                SELECT repo_name
                FROM  `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
                GROUP BY repo_name
            )
        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM 
        repolist as sf 
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON sc.repo_name = sf.repo_name 
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

# check how big this query will be
github.estimate_query_size(query_commit)
python_commits = github.query_to_pandas_safe(query_commit, max_gb_scanned=6)
python_commits