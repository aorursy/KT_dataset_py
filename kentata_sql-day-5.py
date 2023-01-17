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

query = ( """
         -- Select all the columns we want in our joined table
         SELECT L.license, COUNT(sf.path) AS number_of_files
         FROM `bigquery-public-data.github_repos.sample_files` as sf
         -- Table to merge into sample_files
         INNER JOIN `bigquery-public-data.github_repos.licenses` as L
              ON sf.repo_name = L.repo_name -- what column should we joi on?
         GROUP BY L.license
         ORDER BY number_of_files DESC
          """
        )

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(file_count_by_license)
# Question
# How many commits (recorded in the "sample_commits" table) 
# have been made in repos written in the Python programming language? 

github.list_tables()
github.head("sample_commits")
github.head("sample_files")
github.table_schema("sample_files")
github.head("sample_files", selected_columns="symlink_target", num_rows=40)
query_1 = ( """ WITH distinct_repos AS
                (
                    SELECT DISTINCT sf.repo_name
                    FROM `bigquery-public-data.github_repos.sample_files` as sf
                    WHERE sf.path LIKE '%.py'
                )

            SELECT  dr.repo_name, COUNT(sc.commit) AS number_of_commits
            FROM distinct_repos AS dr
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
               ON dr.repo_name = sc.repo_name
            GROUP BY dr.repo_name
            ORDER BY number_of_commits DESC
            """
          )

commits_count_by_repos = github.query_to_pandas_safe(query_1, max_gb_scanned=6)
print(commits_count_by_repos.shape)
print(commits_count_by_repos)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# avoid producing a graph with axis in 1e7 units
plt.ticklabel_format(style='plain')

plt.barh(range(commits_count_by_repos.shape[0]), commits_count_by_repos.number_of_commits.values,
        tick_label=commits_count_by_repos.repo_name)

