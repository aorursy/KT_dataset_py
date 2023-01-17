# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

github_helper = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                        dataset_name = "github_repos")
github_helper.list_tables()
github_helper.head('sample_files')
github_helper.head('licenses')
query = """SELECT L.license, COUNT(sf.path) AS num_files
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.licenses` as L
                ON sf.repo_name = L.repo_name 
            GROUP BY L.license
            ORDER BY num_files DESC
        """

github_helper.estimate_query_size(query)
file_count_by_license = github_helper.query_to_pandas_safe(query, max_gb_scanned=6)
file_count_by_license.head()
len(file_count_by_license)
#github_helper.table_schema('sample_commits')
#github_helper.head('sample_commits')
query2 = """WITH files AS
            (
                SELECT path, repo_name
                FROM `bigquery-public-data.github_repos.sample_files` 
                WHERE path LIKE '%py'
            )
            SELECT sf.repo_name, COUNT(sc.commit) as num_commits
            FROM files AS sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                ON sc.repo_name = sf.repo_name
            GROUP BY sf.repo_name
            ORDER BY num_commits DESC
            """
github_helper.estimate_query_size(query2)
py_commits = github_helper.query_to_pandas_safe(query2, max_gb_scanned=6)
py_commits.head()
len(py_commits)

