# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import bq_helper
gh = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
    SELECT
        repo_name,
        COUNT(*) as countCommits
    FROM
        `bigquery-public-data.github_repos.sample_commits`
    WHERE repo_name IN (
        SELECT
            repo_name
        FROM
            `bigquery-public-data.github_repos.sample_files`
        WHERE
            path LIKE '%.py'
    )
    GROUP BY
        repo_name
    ORDER BY
        countCommits DESC 
"""

x = gh.query_to_pandas(query)
x
x.groupby('repo_name').mean().plot(kind='bar');
# using subquery
query1 = """
    SELECT
        repo_name,
        COUNT(*) as countCommits
    FROM
        `bigquery-public-data.github_repos.sample_commits`
    WHERE repo_name IN (
        SELECT
            repo_name
        FROM
            `bigquery-public-data.github_repos.sample_files`
        WHERE
            path LIKE '%.py'
    )
    GROUP BY
        repo_name
    ORDER BY
        countCommits DESC
"""

query2 = """
    WITH python_repos AS (
    SELECT DISTINCT repo_name -- Notice DISTINCT
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
    SELECT commits.repo_name, COUNT(commit) AS num_commits
    FROM `bigquery-public-data.github_repos.sample_commits` AS commits
    JOIN python_repos
        ON  python_repos.repo_name = commits.repo_name
    GROUP BY commits.repo_name
    ORDER BY num_commits DESC
"""
gh.estimate_query_size(query1), gh.estimate_query_size(query2)
