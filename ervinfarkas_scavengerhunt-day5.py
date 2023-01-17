# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper 
# create a helper object for Hacker News dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="github_repos")

# print a list of all the tables in the Hacker News dataset
github.list_tables()

bitcoin.head('transactions')
# check 'sample_commits' table content
github.table_schema('sample_commits')
# check 'sample_commits' table content
github.table_schema('sample_files')
github.head('sample_files',selected_columns='path')
# USE CTE to get datetime timestamp from integer timestamp and then use select and group by day
query = """ SELECT COUNT(sc.commit) AS commits
            FROM `bigquery-public-data.github_repos.sample_commits` as sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sc.repo_name = sf.repo_name 
            WHERE  sf.path LIKE '%.py'
        """


# check how big this query will be
github.estimate_query_size(query)
# run the query and get transactions by day
commits = github.query_to_pandas_safe(query, max_gb_scanned=6)

print(commits)