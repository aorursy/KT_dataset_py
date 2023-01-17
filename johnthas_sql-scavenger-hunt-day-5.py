# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # BigQuery
github_repos = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='github_repos')
github_repos.list_tables()
github_repos.head('sample_files')
github_repos.head('sample_commits')
query = """
SELECT count(c.commit)
FROM `bigquery-public-data.github_repos.sample_commits` as c
INNER JOIN `bigquery-public-data.github_repos.sample_files` as f
ON c.repo_name = f.repo_name
WHERE f.path LIKE '%.py'
"""

# check query size
size = github_repos.estimate_query_size(query)
size
python_commits = github_repos.query_to_pandas_safe(query, max_gb_scanned=6)
python_commits
