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
import bq_helper

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="github_repos")
github.list_tables()
query = ("""
         -- Choose the columsn we need to in the joined table
         SELECT L.license, COUNT(sf.path) AS number_of_files
         FROM `bigquery-public-data.github_repos.sample_files` as sf
         -- Table to merge into sample_Files
         INNER JOIN `bigquery-public-data.github_repos.licenses` as L
            ON sf.repo_name = L.repo_name 
            GROUP BY L.license
            ORDER BY number_of_files DESC
            """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
file_count_by_license.head(16)
github.head("sample_files")
github.head("sample_commits")
query_py = """
           Select COUNT(sc.repo_name) AS number_of_commits, sf.path
           FROM `bigquery-public-data.github_repos.sample_commits` as sc
           
           INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
               on sc.repo_name = sf.repo_name 
           WHERE sf.path LIKE '%.py'
           GROUP BY sf.path
           ORDER BY number_of_commits DESC
           """
commit_count_py_files = github.query_to_pandas_safe(query_py, max_gb_scanned=6)
print(commit_count_py_files)
query_py2 = """
            SELECT COUNT(sf.id) AS number_of_commits, sf.repo_name
           FROM `bigquery-public-data.github_repos.sample_files` as sf
           INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
               on sf.repo_name = sc.repo_name 
           WHERE sf.path LIKE '%.py'
           GROUP BY sf.repo_name
           ORDER BY number_of_commits DESC
            """
count_py_files = github.query_to_pandas_safe(query_py2, max_gb_scanned=10)
print(count_py_files)