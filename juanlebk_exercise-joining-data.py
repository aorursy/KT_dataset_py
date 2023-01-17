# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("sample_commits")
github.table_schema("sample_repos")
github.table_schema("sample_commits")
github.table_schema("sample_contents")
github.table_schema("sample_files")
github.table_schema("languages")
github.head("languages")
query = """SELECT repo_name,
            COUNT(SC.commit) as commits
            FROM `bigquery-public-data.github_repos.sample_files` AS SF
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS SC
            USING (repo_name)
            WHERE SF.path LIKE '%.py'
            GROUP BY repo_name
            ORDER BY commits DESC"""
github.estimate_query_size(query)
commit_py_count_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=5.3)
commit_py_count_by_repo