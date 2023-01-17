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
import bq_helper

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_files')
#github.head('sample_commits')

query = """
       WITH REPO_TABLE AS
       (
           SELECT DISTINCT repo_name
           FROM `bigquery-public-data.github_repos.sample_files`
           WHERE path LIKE '%.py'
       )
       SELECT COUNT(C.commit) as commit_count, RT.repo_name
           FROM `bigquery-public-data.github_repos.sample_commits` as C
           INNER JOIN REPO_TABLE as RT
           ON C.repo_name = RT.repo_name
       GROUP BY repo_name
       ORDER BY commit_count       
       """

commit_count_data = github.query_to_pandas_safe(query, max_gb_scanned=11)
print(commit_count_data)