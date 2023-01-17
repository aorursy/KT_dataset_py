# import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import our BigQuery helper library
import bq_helper
# instantiate helper object
githelper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name='github_repos')
# check out what tables are available
githelper.list_tables()
# take a look at one of the tables:
githelper.head('commits')
# build a query using JOIN
query = """
    WITH python_repos AS (
    SELECT DISTINCT repo_name -- Notice DISTINCT
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
    
    SELECT COUNT(C.commit) as commits, C.repo_name as repo
    FROM `bigquery-public-data.github_repos.sample_commits` as C
    INNER JOIN python_repos
        ON C.repo_name = python_repos.repo_name
    GROUP BY repo
    ORDER BY commits DESC
    """

# estimate the query's usage
githelper.estimate_query_size(query)
# run that suckah!
pycommits = githelper.query_to_pandas_safe(query, max_gb_scanned=6)
# check out the results
pycommits
sorted = pycommits.set_index('repo').sort_values('commits', ascending=False)
sorted
plt.style.use('ggplot')
sorted.plot(kind='bar', figsize=(12,12))
plt.title("No. of Commits Written in Python by Repo")
plt.xticks(rotation=45)
