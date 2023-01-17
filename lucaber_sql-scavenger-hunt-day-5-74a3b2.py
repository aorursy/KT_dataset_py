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
github.head('sample_files')
github.head('sample_commits')
query='''
        WITH joined_mat AS
        (
          SELECT sf.path AS path,
              sc.commit AS commit
          FROM `bigquery-public-data.github_repos.sample_files` as sf
          INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
          ON sf.repo_name = sc.repo_name
        )
        SELECT path, COUNT(commit) AS total_commits
        FROM joined_mat
        GROUP BY path
        HAVING path LIKE '%.py'
        ORDER BY COUNT(commit) DESC
      '''
commits_per_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
commits_per_repo.head()