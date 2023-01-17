# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = (""" 
        WITH GoodRepo as (SELECT repo_name 
        FROM `bigquery-public-data.github_repos.sample_files` 
        WHERE path like '%.py'
        group by repo_name)
        
        SELECT sc.repo_name, count(commit)
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN GoodRepo as gr
        ON 
        gr.repo_name = sc.repo_name
        group by sc.repo_name
        """)

res = github.query_to_pandas_safe(query, max_gb_scanned=6)
res
# You can use two dashes (--) to add comments in SQL
query = (""" 

        """)

res = github.query_to_pandas_safe(query, max_gb_scanned=6)
res
github.list_tables()
github.table_schema("sample_files")
github.table_schema("sample_commits")
# You can use two dashes (--) to add comments in SQL
# You can use two dashes (--) to add comments in SQL
query = (""" 
        SELECT repo_name, count(path)
        FROM `bigquery-public-data.github_repos.sample_files` 
        WHERE path like '%.py'
        group by repo_name
        """)

res = github.query_to_pandas_safe(query, max_gb_scanned=6)
res