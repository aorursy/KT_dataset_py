# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# import package with helper functions
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="github_repos")
github.list_tables()
github.head("sample_files")
# you can use two dashes (--) to add comments in SQL
query = """ 
        -- Slect all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license 
        ORDER BY number_of_files DESC

"""
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
github.head("commits")
github.head("sample_repos")
github.head("sample_files")
github.head("sample_commits")
query2 = """ 
        -- Slect all the columns we want in our joined table
        SELECT sf.repo_name as repo, COUNT(sc.commit) AS commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        -- Table to merge into sample files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY repo
        ORDER BY commits DESC

"""
python_commits = github.query_to_pandas_safe(query2, max_gb_scanned=6)
print(python_commits)
import matplotlib.pyplot as plt
import numpy as np

plt.pie(python_commits.commits, labels=python_commits.repo, autopct='%1.1f%%', shadow=True)
plt.show()
