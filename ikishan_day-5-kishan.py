# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = ("""
              SELECT sc.repo_name, COUNT (sc.commit) AS Number_OF_commits_py
                  FROM `bigquery-public-data.github_repos.sample_commits` AS sc
               INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
               ON sf.repo_name = sc.repo_name
                WHERE sf.path LIKE '%.py'
                  GROUP BY sc.repo_name
                  ORDER BY Number_OF_commits_py DESC
                  """)
No_of_py = github.query_to_pandas_safe(query,max_gb_scanned = 7)
print(No_of_py)