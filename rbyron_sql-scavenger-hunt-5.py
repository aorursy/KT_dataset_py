# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head('sample_files', 2)
github.head('sample_commits', 2)
query = ("""WITH py_repos AS (
                --finds repos that contain python files
                SELECT DISTINCT repo_name 
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%py'
            )
                
            --find number of commits made in py_repos
            SELECT commits.repo_name, COUNT(commits.commit) as number_of_commits
            FROM `bigquery-public-data.github_repos.sample_commits` as commits
            INNER JOIN py_repos ON py_repos.repo_name = commits.repo_name
            GROUP BY repo_name
            ORDER BY number_of_commits DESC
         """)

commit_count_py = github.query_to_pandas_safe(query, max_gb_scanned=7)
commit_count_py.head()