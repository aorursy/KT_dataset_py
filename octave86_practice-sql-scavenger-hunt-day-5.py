# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

github.list_tables()

github.table_schema("files")
query = """SELECT L.license, COUNT(sf.path) AS number_of_files FROM `bigquery-public-data.github_repos.sample_files` AS sf 
INNER JOIN `bigquery-public-data.github_repos.licenses`AS L ON sf.repo_name=L.repo_name GROUP BY L.license ORDER BY number_of_files DESC """

github.estimate_query_size(query)
file_count_by_license = github.query_to_pandas_safe(query,max_gb_scanned=6)
print(file_count_by_license)
query1="""SELECT COUNT(sc.commit) AS no_of_commits, sf.repo_name AS Repo FROM
`bigquery-public-data.github_repos.sample_files` AS sf INNER JOIN
`bigquery-public-data.github_repos.sample_commits` AS sc ON sf.repo_name=sc.repo_name
WHERE sf.path LIKE '%.py' GROUP BY Repo ORDER BY no_of_commits DESC
"""

github.estimate_query_size(query1)
commit_count_by_repos=github.query_to_pandas_safe(query1,max_gb_scanned=6)
print(commit_count_by_repos)
