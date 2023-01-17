# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = '''select count(sc.commit) as py_commits, sc.repo_name
           from `bigquery-public-data.github_repos.sample_commits` as sc
           inner join `bigquery-public-data.github_repos.sample_files` as sf on sf.repo_name = sc.repo_name
           where sf.path like "%.py"
           group by sc.repo_name
           order by py_commits desc'''
github.estimate_query_size(query)
total_py_repos = github.query_to_pandas_safe(query, max_gb_scanned=6)
total_py_repos