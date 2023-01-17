# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
print(github.table_schema('sample_commits').to_string())
print(github.table_schema('sample_files').to_string())
df=github.query_to_pandas("""
SELECT COUNT(commit) FROM `bigquery-public-data.github_repos.sample_commits` sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` sf ON sc.repo_name=sf.repo_name
WHERE sf.path like "%.py"
""")
df