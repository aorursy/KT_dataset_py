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
github.head("sample_commits")
github.head("sample_files")
# Your code goes here :)
query2 = """select sf.repo_name, count(sc.commit) as total_commits
            from `bigquery-public-data.github_repos.sample_commits` as sc
            inner join `bigquery-public-data.github_repos.sample_files` as sf
                on sc.repo_name = sf.repo_name
            where sf.path like '%.py'
            group by sf.repo_name
            order by total_commits desc
            """

github.estimate_query_size(query2)
commits_by_repo = github.query_to_pandas_safe(query2, max_gb_scanned = 6)
commits_by_repo.shape
commits_by_repo