import numpy as np # linear algebra
# pandas for handling data
import pandas as pd
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# use fivethirtyeight style for beautiful plot
plt.style.use('fivethirtyeight')
QUERY = """
    SELECT
        -- calculate the count of commit column in sample_commit table
        COUNT(sample_commit.commit) as count_of_commit_using_python
    FROM
      `bigquery-public-data.github_repos.sample_commits` AS sample_commit
      -- inner join with sample_files table with condition sample_commit.repo_name = sample_files.repo_name
    INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sample_files
        ON sample_commit.repo_name = sample_files.repo_name
    WHERE
        -- where samples_files path column has .py extension
      sample_files.path LIKE '%.py'
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
# I add a parametre called max_gb_scanned = 6 . This query size is more than 1 GB
df_commit_using_python_count = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=6)
df_commit_using_python_count
