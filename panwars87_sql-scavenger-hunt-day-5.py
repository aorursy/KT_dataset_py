import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
    select sc.repo_name, count(distinct commit) as commit_count
    from `bigquery-public-data.github_repos.sample_commits` as sc
    inner join `bigquery-public-data.github_repos.sample_files` as sf
    on sc.repo_name = sf.repo_name and sf.path like '%.py'
    group by sc.repo_name;
"""

query_ds = github.query_to_pandas_safe(query, max_gb_scanned=6)
query_ds.head()
import matplotlib.pyplot as plt

plt.subplots(figsize=(12,5))
plt.bar(query_ds.repo_name, query_ds.commit_count)
plt.title("Commit per pythong repo")
